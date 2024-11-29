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
            const useSequential = event.data.useSequential;
            const sparePlaceholder = event.data.sparePlaceholder;

            if (!MLMModel.maskToken) {
                throw new Error('Mask token not available. Please check model documentation.');
            }

            // Normalize text: replace triple or more spaces with double space
            // and normalize other whitespace (tabs, newlines) to single space
            let normalizedText = text
                .replace(/\s{3,}/g, '  ')  // Replace 3+ spaces with double space
                .replace(/[^\S ]+/g, ' ')  // Replace other whitespace (not spaces) with single space
                .trim();
            
            // Find all mask patterns and their positions
            const masks = [];
            const patterns = ['??', '<mask>', '[MASK]'];
            
            // Find all mask positions with their original text
            patterns.forEach(pattern => {
                let pos = 0;
                let tempText = normalizedText;
                
                while ((pos = tempText.indexOf(pattern)) !== -1) {
                    const globalPos = normalizedText.indexOf(pattern, masks.length > 0 ? masks[masks.length - 1].end : 0);
                    masks.push({
                        start: globalPos,
                        end: globalPos + pattern.length,
                        pattern: pattern,
                        length: pattern.length,
                        originalText: pattern
                    });
                    tempText = tempText.substring(pos + pattern.length);
                }
            });

            // Sort masks by position
            masks.sort((a, b) => a.start - b.start);

            if (masks.length === 0) {
                postMessage({
                    status: 'error',
                    predictions: [`Text must contain <mask> or ?? (model uses ${MLMModel.maskToken} internally)`]
                });
                return;
            }

            const startTime = performance.now();
            const allPredictions = [];
            let currentText = normalizedText;

            // Add a helper function to log the exact text being used
            const logInferenceText = (step, text) => {
                const textForDisplay = text.split('').map(char => {
                    if (char === ' ') return '␣';  // Unicode symbol for space
                    return char;
                }).join('');
                console.log(`Inference #${step} exact text: "${textForDisplay}"`);
                console.log(`Inference #${step} raw text: "${text}"`);
            };

            // Process each mask position separately
            for (let i = 0; i < masks.length; i++) {
                let currentMaskedText = currentText;
                let positionAdjustment = 0;
                
                masks.forEach((mask, index) => {
                    const adjustedStart = mask.start + positionAdjustment;
                    const adjustedEnd = mask.end + positionAdjustment;
                    
                    if (index === i) {
                        // Replace current mask with model's mask token
                        currentMaskedText = 
                            currentMaskedText.substring(0, adjustedStart) + 
                            MLMModel.maskToken + 
                            currentMaskedText.substring(adjustedEnd);
                        positionAdjustment += MLMModel.maskToken.length - mask.length;
                    } else if (index > i) {
                        // Replace future masks with spare placeholder
                        const placeholder = sparePlaceholder || MLMModel.maskToken;
                        currentMaskedText = 
                            currentMaskedText.substring(0, adjustedStart) + 
                            placeholder + 
                            currentMaskedText.substring(adjustedEnd);
                        positionAdjustment += placeholder.length - mask.length;
                    }
                });

                // Log the exact text being used for inference
                logInferenceText(i + 1, currentMaskedText);

                console.log(`Inference #${i + 1} using text:`, currentMaskedText);

                const outputs = await mlm(currentMaskedText, {
                    topk: 5,
                });

                const predictions = outputs.map(output => {
                    const decodedToken = MLMModel.tokenizer.decode([output.token], {
                        skip_special_tokens: true,
                        clean_up_tokenization_spaces: true
                    }).trim();
                    return {
                        text: decodedToken,
                        score: output.score,
                        display: `${decodedToken} (${(output.score * 100).toFixed(2)}%)`
                    };
                });

                allPredictions.push({
                    maskIndex: i + 1,
                    originalPattern: masks[i].pattern,
                    position: masks[i].start,
                    predictions: predictions.map(p => p.display),
                    inferenceText: currentMaskedText,
                    inferenceTextDisplay: currentMaskedText.split('').map(char => 
                        char === ' ' ? '␣' : char
                    ).join('')
                });

                // If using sequential unmasking, update the text with the best prediction
                if (useSequential && i < masks.length - 1) {
                    const bestPrediction = predictions[0].text;
                    currentText = 
                        currentText.substring(0, masks[i].start) + 
                        bestPrediction + 
                        currentText.substring(masks[i].start + masks[i].length);
                }
            }

            const inferenceTime = performance.now() - startTime;

            postMessage({
                status: 'complete',
                allPredictions,
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