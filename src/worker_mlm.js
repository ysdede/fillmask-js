import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

class MLMModel {
    static pipeline = null;
    static tokenizer = null;
    static maskToken = null;

    static async tryLoadPipeline(modelName, deviceType, quantization, progress_callback) {
        try {
            return await pipeline('fill-mask', modelName, {
                progress_callback: (progress) => {
                    if (progress && progress.status === 'progress' && progress.total > 0) {
                        const percentage = Math.min(Math.round((progress.loaded / progress.total) * 100), 100);
                        const loaded = (progress.loaded / (1024 * 1024)).toFixed(1);
                        const total = (progress.total / (1024 * 1024)).toFixed(1);
                        postMessage({
                            status: 'progress',
                            message: `Downloading model: ${percentage}% (${loaded}MB / ${total}MB)`
                        });
                    }
                },
                quantized: quantization !== 'fp32',
                dtype: quantization,
                device: deviceType
            });
        } catch (error) {
            console.warn(`Failed to load with ${quantization}:`, error.message);
            return null;
        }
    }

    static async warmupInference(pipeline) {
        postMessage({
            status: 'progress',
            message: 'Warming up model...'
        });

        try {
            // Use the model's actual mask token
            const dummyText = `This is a ${pipeline.tokenizer.mask_token} sentence.`;
            await pipeline(dummyText, {
                topk: 1,
            });
        } catch (error) {
            console.warn('Warmup inference failed:', error);
            // Continue even if warmup fails
        }
    }

    static async getInstance(modelName, deviceType, quantization, progress_callback = null) {
        if (!this.pipeline) {
            // Try the requested quantization first
            this.pipeline = await this.tryLoadPipeline(modelName, deviceType, quantization, progress_callback);

            // If failed, try the default fallback based on device
            if (!this.pipeline) {
                const fallbackType = deviceType === 'webgpu' ? 'fp32' : 'q8';
                
                if (quantization !== fallbackType) {
                    postMessage({
                        status: 'warning',
                        message: `Falling back to ${fallbackType} for ${deviceType}`,
                        duration: 3000
                    });

                    this.pipeline = await this.tryLoadPipeline(modelName, deviceType, fallbackType, progress_callback);
                }
            }

            if (!this.pipeline) {
                throw new Error(`Failed to load model. Please try a different model or device.`);
            }
            
            this.tokenizer = this.pipeline.tokenizer;

            if (this.tokenizer.mask_token) {
                this.maskToken = this.tokenizer.mask_token;
                console.log(`Found mask token in tokenizer: ${this.maskToken}`);

                // Perform warmup inference after successful model load
                await this.warmupInference(this.pipeline);
            } else {
                console.warn('Could not detect mask token automatically');
                postMessage({
                    status: 'warning',
                    message: 'Could not detect mask token automatically. Please check model documentation.',
                    duration: 5000
                });
                return null;
            }
        }
        return this.pipeline;
    }
}

self.onmessage = async (event) => {
    try {
        const { type, text, model, device, quantization } = event.data;

        if (type === 'load_model') {
            postMessage({ status: 'loading', message: 'Starting model download...' });

            const startTime = performance.now();
            try {
                const pipeline = await MLMModel.getInstance(model, device, quantization);
                if (!pipeline) {
                    throw new Error('Failed to initialize model pipeline');
                }
                const loadTime = performance.now() - startTime;

                postMessage({ 
                    status: 'model_loaded',
                    loadTime: loadTime.toFixed(0),
                    maskToken: MLMModel.maskToken
                });
            } catch (error) {
                if (device === 'webgpu' && error.message.includes('WebGPU')) {
                    postMessage({
                        status: 'error',
                        predictions: ['WebGPU is not available. Please try using WASM backend instead.']
                    });
                } else {
                    throw error;
                }
            }
            return;
        }

        if (type === 'predict') {
            const mlm = await MLMModel.getInstance();

            if (!MLMModel.maskToken) {
                throw new Error('Mask token not available. Please check model documentation.');
            }

            const processedText = text
                .replace(/\.\./g, MLMModel.maskToken)
                .replace(/<mask>/g, MLMModel.maskToken)
                .replace(/\[MASK\]/g, MLMModel.maskToken);

            if (!processedText.includes(MLMModel.maskToken)) {
                postMessage({
                    status: 'error',
                    predictions: [`Text must contain <mask> or .. (model uses ${MLMModel.maskToken} internally)`]
                });
                return;
            }

            const startTime = performance.now();
            const outputs = await mlm(processedText, {
                topk: 5,
            });
            const inferenceTime = performance.now() - startTime;

            const predictions = outputs.map(output => {
                const decodedToken = MLMModel.tokenizer.decode([output.token], {
                    skip_special_tokens: true,
                    clean_up_tokenization_spaces: true
                }).trim();
                return `${decodedToken} (${(output.score * 100).toFixed(2)}%)`;
            });

            postMessage({
                status: 'complete',
                predictions,
                inferenceTime: inferenceTime.toFixed(0)
            });
        }
    } catch (error) {
        console.error('Error in worker:', error);
        postMessage({
            status: 'error',
            predictions: [`Error: ${error.message}`]
        });
    }
}; 