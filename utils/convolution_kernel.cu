__global__ void convolve(const unsigned char *input_image, unsigned char *output_image,
                                          const float * __restrict__ kernel, int width, int height,
                                          int channels, int kernel_size) {
    // Calculate the pixel's location
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // half of the kernel size
    int padding = max(1, kernel_size / 2);

    // Make sure we don't go out of bounds
    if (x >= padding && y >= padding && x < (width - padding) && y < (height - padding)) {
        // For each color channel
        for (int ch = 0; ch < channels; ch++) {
            float channel_sum = 0.0f;
            for (int ky = -padding; ky <= padding; ky++) {
                for (int kx = -padding; kx <= padding; kx++) {
                    // Get the kernel value at location (kx, ky)
                    float kernel_value = kernel[(ky + padding) * kernel_size + (kx + padding)];
                    // Get the image value at location (x + kx, y + ky) if given color channel and multiply by the kernel value
                    channel_sum += input_image[((y + ky) * width + (x + kx)) * channels + ch] * kernel_value;
                }
            }
            // Set the output image value at location (x, y) for given color channel, and clamp to [0, 255]
            output_image[(y * width + x) * channels + ch] = min(max(int(channel_sum), 0), 255);
        }
    }
}

// An attempt at using shared memory to speed up convolution, but I didn't observe any speedup
__global__ void convolveShared(const unsigned char *input_image, unsigned char *output_image,
                               const float * __restrict__ kernel, int width, int height,
                               int channels, int kernel_size) {
    // Calculate necessary padding around blocks
    const int padding = max(1, kernel_size / 2);
    const int blockSizeX = blockDim.x + 2 * padding; // Block width in shared memory
    const int blockSizeY = blockDim.y + 2 * padding; // Block height in shared memory

    // Shared memory allocation
    extern __shared__ unsigned char sharedBlock[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + padding;
    int ty = threadIdx.y + padding;

    // Load data into shared memory
    for (int ch = 0; ch < channels; ch++) {
        // Main area
        if (x < width && y < height) {
            sharedBlock[(ty * blockSizeX + tx) * channels + ch] = input_image[(y * width + x) * channels + ch];
        }

        // Load top & bottom padding
        if (threadIdx.y < padding) {
            // Top padding
            if (blockIdx.y > 0 && y - padding < height) {
                sharedBlock[((ty - padding) * blockSizeX + tx) * channels + ch] = input_image[((y - padding) * width + x) * channels + ch];
            }
            // Bottom padding
            if (blockIdx.y < gridDim.y - 1 && y + blockDim.y < height) {
                sharedBlock[((ty + blockDim.y) * blockSizeX + tx) * channels + ch] = input_image[((y + blockDim.y) * width + x) * channels + ch];
            }
        }

        // Load left & right padding
        if (threadIdx.x < padding) {
            // Left padding
            if (blockIdx.x > 0 && x - padding < width) {
                sharedBlock[(ty * blockSizeX + tx - padding) * channels + ch] = input_image[(y * width + x - padding) * channels + ch];
            }
            // Right padding
            if (blockIdx.x < gridDim.x - 1 && x + blockDim.x < width) {
                sharedBlock[(ty * blockSizeX + tx + blockDim.x) * channels + ch] = input_image[(y * width + x + blockDim.x) * channels + ch];
            }
        }
    }

    __syncthreads();

    // Perform convolution using shared memory
    if (x < width && y < height) {
        for (int ch = 0; ch < channels; ch++) {
            float channel_sum = 0.0f;
            for (int ky = -padding; ky <= padding; ky++) {
                for (int kx = -padding; kx <= padding; kx++) {
                    float kernel_value = kernel[(ky + padding) * kernel_size + (kx + padding)];
                    int sharedX = tx + kx;
                    int sharedY = ty + ky;
                    channel_sum += sharedBlock[(sharedY * blockSizeX + sharedX) * channels + ch] * kernel_value;
                }
            }
            // Set the output image value
            output_image[(y * width + x) * channels + ch] = min(max(int(channel_sum), 0), 255);
        }
    }
}