#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define INPUT_SIZE 3
#define HIDDEN_SIZE 16s
#define OUTPUT_SIZE 3
#define LEARNING_RATE 0.01f
#define BATCH_SIZE 16
#define ITERATIONS 5000

// Xavier initialization (host side)
__host__ void xavier_init(float* arr, int in_dim, int out_dim) {
    float scale = sqrtf(6.0f / (in_dim + out_dim));
    for (int i = 0; i < in_dim * out_dim; ++i) {
        arr[i] = ((rand() / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float s) {
    return s * (1.0f - s);
}

// Kernel to write a fixed value into an array (used to zero gradients)
__global__ void initialize(float* arr, int size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) arr[i] = value;
}

// Randomly sample (x,y,t) ∈ [0,2π]×[0,2π]×[0,1] and compute the exact Taylor-Green solution
__global__ void init_data(float* inputs,
    float* targets_u, float* targets_v, float* targets_p,
    curandState* states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < BATCH_SIZE) {
        curand_init(1234, idx, 0, &states[idx]);

        float x = curand_uniform(&states[idx]) * 2.0f * M_PI;
        float y = curand_uniform(&states[idx]) * 2.0f * M_PI;
        float t = curand_uniform(&states[idx]);

        inputs[idx * INPUT_SIZE + 0] = x;
        inputs[idx * INPUT_SIZE + 1] = y;
        inputs[idx * INPUT_SIZE + 2] = t;

        // Exact Taylor-Green vortex solution
        targets_u[idx] = -cosf(x) * sinf(y) * expf(-2 * t);
        targets_v[idx] = sinf(x) * cosf(y) * expf(-2 * t);
        targets_p[idx] = -0.25f * (cosf(2 * x) + cosf(2 * y)) * expf(-4 * t);
    }
}

// Forward and backward pass for one mini-batch (each thread handles one sample)
__global__ void forward_and_backward(
    float* inputs,
    float* targets_u, float* targets_v, float* targets_p,
    float* W1, float* b1, float* W2, float* b2,
    float* dW1, float* db1, float* dW2, float* db2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BATCH_SIZE) return;

    // 1) Load one sample's input
    float x_in[INPUT_SIZE];
    for (int j = 0; j < INPUT_SIZE; ++j) {
        x_in[j] = inputs[idx * INPUT_SIZE + j];
    }
    float u_true = targets_u[idx];
    float v_true = targets_v[idx];
    float p_true = targets_p[idx];

    // 2) Forward pass: Hidden layer
    float a1[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        float sum = b1[h];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            sum += W1[h * INPUT_SIZE + j] * x_in[j];
        }
        a1[h] = sigmoid(sum);
    }

    // 3) Forward pass: Output layer
    float u_pred = b2[0];
    float v_pred = b2[1];
    float p_pred = b2[2];
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        u_pred += W2[0 * HIDDEN_SIZE + h] * a1[h];
        v_pred += W2[1 * HIDDEN_SIZE + h] * a1[h];
        p_pred += W2[2 * HIDDEN_SIZE + h] * a1[h];
    }

    // 4) Compute differences (MSE)
    float du = u_pred - u_true;
    float dv = v_pred - v_true;
    float dp = p_pred - p_true;

    // 5) Backpropagation: Output layer gradients
    float d_out[OUTPUT_SIZE] = { 2.0f * du, 2.0f * dv, 2.0f * dp };
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            // dW2[o,h] += d_out[o] * a1[h]
            atomicAdd(&dW2[o * HIDDEN_SIZE + h], d_out[o] * a1[h]);
        }
        // db2[o] += d_out[o]
        atomicAdd(&db2[o], d_out[o]);
    }

    // 6) Backpropagation: Hidden layer gradients
    float dz1[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        // Sum contributions from all output nodes
        float grad = d_out[0] * W2[0 * HIDDEN_SIZE + h]
            + d_out[1] * W2[1 * HIDDEN_SIZE + h]
            + d_out[2] * W2[2 * HIDDEN_SIZE + h];
        // Multiply by sigmoid′(a1[h])
        dz1[h] = grad * sigmoid_derivative(a1[h]);

        for (int j = 0; j < INPUT_SIZE; ++j) {
            // dW1[h,j] += dz1[h] * x_in[j]
            atomicAdd(&dW1[h * INPUT_SIZE + j], dz1[h] * x_in[j]);
        }
        // db1[h] += dz1[h]
        atomicAdd(&db1[h], dz1[h]);
    }
}

// Update a parameter array by gradient descent: arr -= lr * darr / BATCH_SIZE
__global__ void update_params(float* arr, float* darr, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] -= lr * darr[idx] / BATCH_SIZE;
    }
}

int main() {
    // Allocate GPU memory
    float* d_inputs, * d_targets_u, * d_targets_v, * d_targets_p;
    float* d_W1, * d_b1, * d_W2, * d_b2;
    float* d_dW1, * d_db1, * d_dW2, * d_db2;
    curandState* d_states;

    cudaMalloc(&d_inputs, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_targets_u, BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_targets_v, BATCH_SIZE * sizeof(float));
    cudaMalloc(&d_targets_p, BATCH_SIZE * sizeof(float));

    cudaMalloc(&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(float));

    cudaMalloc(&d_dW1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_db1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_dW2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_db2, OUTPUT_SIZE * sizeof(float));

    cudaMalloc(&d_states, BATCH_SIZE * sizeof(curandState));

    // Xavier initialization on host, then copy to GPU
    float h_W1[HIDDEN_SIZE * INPUT_SIZE];
    float h_b1[HIDDEN_SIZE];
    float h_W2[OUTPUT_SIZE * HIDDEN_SIZE];
    float h_b2[OUTPUT_SIZE];

    xavier_init(h_W1, INPUT_SIZE, HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; ++i) h_b1[i] = 0.0f;
    xavier_init(h_W2, HIDDEN_SIZE, OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; ++i) h_b2[i] = 0.0f;

    cudaMemcpy(d_W1, h_W1, sizeof(h_W1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, sizeof(h_b1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, sizeof(h_W2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, sizeof(h_b2), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;

    // Training loop
    for (int epoch = 0; epoch < ITERATIONS; ++epoch) {
        // Zero out gradient buffers
        initialize << <(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256 >> > (d_dW1, HIDDEN_SIZE * INPUT_SIZE, 0.0f);
        initialize << <(HIDDEN_SIZE + 255) / 256, 256 >> > (d_db1, HIDDEN_SIZE, 0.0f);
        initialize << <(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256 >> > (d_dW2, OUTPUT_SIZE * HIDDEN_SIZE, 0.0f);
        initialize << <(OUTPUT_SIZE + 255) / 256, 256 >> > (d_db2, OUTPUT_SIZE, 0.0f);

        // 1) Sample a new mini-batch & compute ground-truth
        init_data << <blocks, threads >> > (d_inputs,
            d_targets_u, d_targets_v, d_targets_p,
            d_states);

        // 2) Forward + backward pass (compute gradients)
        forward_and_backward << <blocks, threads >> > (d_inputs,
            d_targets_u, d_targets_v, d_targets_p,
            d_W1, d_b1, d_W2, d_b2,
            d_dW1, d_db1, d_dW2, d_db2);

        // 3) Update weights & biases
        update_params << <(HIDDEN_SIZE * INPUT_SIZE + 255) / 256, 256 >> > (d_W1, d_dW1, LEARNING_RATE, HIDDEN_SIZE * INPUT_SIZE);
        update_params << <(HIDDEN_SIZE + 255) / 256, 256 >> > (d_b1, d_db1, LEARNING_RATE, HIDDEN_SIZE);
        update_params << <(OUTPUT_SIZE * HIDDEN_SIZE + 255) / 256, 256 >> > (d_W2, d_dW2, LEARNING_RATE, OUTPUT_SIZE * HIDDEN_SIZE);
        update_params << <(OUTPUT_SIZE + 255) / 256, 256 >> > (d_b2, d_db2, LEARNING_RATE, OUTPUT_SIZE);

        // 4) Every 500 epochs, compute and print average loss for current batch
        if ((epoch + 1) % 500 == 0) {
            // Copy current batch inputs & parameters back to host
            float h_inputs[BATCH_SIZE * INPUT_SIZE];
            cudaMemcpy(h_inputs, d_inputs, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            float h_W1_copy[HIDDEN_SIZE * INPUT_SIZE], h_b1_copy[HIDDEN_SIZE];
            float h_W2_copy[OUTPUT_SIZE * HIDDEN_SIZE], h_b2_copy[OUTPUT_SIZE];
            cudaMemcpy(h_W1_copy, d_W1, sizeof(h_W1_copy), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_b1_copy, d_b1, sizeof(h_b1_copy), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_W2_copy, d_W2, sizeof(h_W2_copy), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_b2_copy, d_b2, sizeof(h_b2_copy), cudaMemcpyDeviceToHost);

            float total_loss = 0.0f;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                float x = h_inputs[i * 3 + 0];
                float y = h_inputs[i * 3 + 1];
                float t = h_inputs[i * 3 + 2];

                // CPU forward pass for this sample
                float hidden[HIDDEN_SIZE];
                for (int h = 0; h < HIDDEN_SIZE; ++h) {
                    float sum = h_b1_copy[h];
                    sum += h_W1_copy[h * INPUT_SIZE + 0] * x;
                    sum += h_W1_copy[h * INPUT_SIZE + 1] * y;
                    sum += h_W1_copy[h * INPUT_SIZE + 2] * t;
                    hidden[h] = 1.0f / (1.0f + expf(-sum));
                }
                float u_pred = h_b2_copy[0];
                float v_pred = h_b2_copy[1];
                float p_pred = h_b2_copy[2];
                for (int h = 0; h < HIDDEN_SIZE; ++h) {
                    u_pred += h_W2_copy[0 * HIDDEN_SIZE + h] * hidden[h];
                    v_pred += h_W2_copy[1 * HIDDEN_SIZE + h] * hidden[h];
                    p_pred += h_W2_copy[2 * HIDDEN_SIZE + h] * hidden[h];
                }

                float u_true = -cosf(x) * sinf(y) * expf(-2 * t);
                float v_true = sinf(x) * cosf(y) * expf(-2 * t);
                float p_true = -0.25f * (cosf(2 * x) + cosf(2 * y)) * expf(-4 * t);

                float du = u_pred - u_true;
                float dv = v_pred - v_true;
                float dp = p_pred - p_true;
                total_loss += du * du + dv * dv + dp * dp;
            }
            printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / BATCH_SIZE);
        }
    }

    // After training, print final predictions vs. true values for last batch
    float h_inputs[BATCH_SIZE * INPUT_SIZE];
    cudaMemcpy(h_inputs, d_inputs, BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    float h_W1_final[HIDDEN_SIZE * INPUT_SIZE], h_b1_final[HIDDEN_SIZE];
    float h_W2_final[OUTPUT_SIZE * HIDDEN_SIZE], h_b2_final[OUTPUT_SIZE];
    cudaMemcpy(h_W1_final, d_W1, sizeof(h_W1_final), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b1_final, d_b1, sizeof(h_b1_final), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2_final, d_W2, sizeof(h_W2_final), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b2_final, d_b2, sizeof(h_b2_final), cudaMemcpyDeviceToHost);

    printf("\nFinal predictions vs. true values for last mini-batch:\n");
    for (int i = 0; i < BATCH_SIZE; ++i) {
        float x = h_inputs[i * 3 + 0];
        float y = h_inputs[i * 3 + 1];
        float t = h_inputs[i * 3 + 2];

        // CPU forward pass one sample
        float hidden[HIDDEN_SIZE];
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            float sum = h_b1_final[h];
            sum += h_W1_final[h * INPUT_SIZE + 0] * x;
            sum += h_W1_final[h * INPUT_SIZE + 1] * y;
            sum += h_W1_final[h * INPUT_SIZE + 2] * t;
            hidden[h] = 1.0f / (1.0f + expf(-sum));
        }
        float u_pred = h_b2_final[0];
        float v_pred = h_b2_final[1];
        float p_pred = h_b2_final[2];
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            u_pred += h_W2_final[0 * HIDDEN_SIZE + h] * hidden[h];
            v_pred += h_W2_final[1 * HIDDEN_SIZE + h] * hidden[h];
            p_pred += h_W2_final[2 * HIDDEN_SIZE + h] * hidden[h];
        }

        float u_true = -cosf(x) * sinf(y) * expf(-2 * t);
        float v_true = sinf(x) * cosf(y) * expf(-2 * t);
        float p_true = -0.25f * (cosf(2 * x) + cosf(2 * y)) * expf(-4 * t);

        printf("Sample %d: (x=%.3f, y=%.3f, t=%.3f)\n", i, x, y, t);
        printf("  u_pred = %.5f , u_true = %.5f\n", u_pred, u_true);
        printf("  v_pred = %.5f , v_true = %.5f\n", v_pred, v_true);
        printf("  p_pred = %.5f , p_true = %.5f\n", p_pred, p_true);
    }

    // Free all GPU memory
    cudaFree(d_inputs);
    cudaFree(d_targets_u);
    cudaFree(d_targets_v);
    cudaFree(d_targets_p);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_dW1);
    cudaFree(d_db1);
    cudaFree(d_dW2);
    cudaFree(d_db2);
    cudaFree(d_states);

    return 0;
}
