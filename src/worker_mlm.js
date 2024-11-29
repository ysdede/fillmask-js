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

// Replace the current mask finding code with this improved version
const findMasks = (text, maskToken) => {
    // Split the text into chunks using the mask token as delimiter
    const chunks = text.split(maskToken);
    
    // If there's only one chunk, no masks were found
    if (chunks.length === 1) {
        return {
            chunks: [text],
            masks: [],
            textArray: [text]
        };
    }

    const masks = [];
    let position = 0;
    const textArray = [];

    // Process each split position to get mask information
    for (let i = 0; i < chunks.length; i++) {
        // Add the text chunk to our array
        textArray.push(chunks[i]);
        
        // If this isn't the last chunk, add a mask
        if (i < chunks.length - 1) {
            position += chunks[i].length;
            
            masks.push({
                start: position,
                end: position + maskToken.length,
                pattern: maskToken,
                length: maskToken.length,
                originalText: maskToken,
                precedingText: chunks[i],
                followingText: chunks[i + 1],
                chunkIndex: i // Store which chunk this mask follows
            });
            
            // Add the mask to our text array
            textArray.push(maskToken);
            
            position += maskToken.length;
        }
    }

    return {
        chunks,      // Original chunks from split
        masks,       // Mask information
        textArray    // Alternating array of [chunk, mask, chunk, mask, chunk]
    };
};

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

            let normalizedText = text.replace(/\s{2,}/g, ' ').trim();
            normalizedText = normalizedText.replaceAll('??', MLMModel.maskToken);
            console.log('Normalized text:', normalizedText);
            
            const { chunks, masks, textArray } = findMasks(normalizedText, MLMModel.maskToken);
            
            if (masks.length === 0) {
                postMessage({
                    status: 'error',
                    predictions: [`Text must contain <mask> or ?? (model uses ${MLMModel.maskToken} internally)`]
                });
                return;
            }

            console.log('Text chunks:', chunks);
            console.log('Masks:', masks);

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

            // Process each mask
            for (let i = 0; i < masks.length; i++) {
                // Reconstruct text with current mask token and placeholders
                let currentMaskedText = '';
                
                for (let j = 0; j < textArray.length; j++) {
                    if (j % 2 === 0) {
                        // Text chunk
                        currentMaskedText += textArray[j];
                    } else {
                        // Mask position
                        const maskIndex = Math.floor(j/2);
                        if (maskIndex === i) {
                            // This is our current mask position
                            currentMaskedText += MLMModel.maskToken;
                        } else if (maskIndex < i) {
                            // Previous masks - use their predictions
                            const previousPrediction = allPredictions[maskIndex].predictions[0].split(' (')[0];
                            currentMaskedText += previousPrediction;
                        } else {
                            // Future masks - use placeholder
                            currentMaskedText += (sparePlaceholder || MLMModel.maskToken);
                        }
                    }
                }

                // Log the exact text being used for inference
                logInferenceText(i + 1, currentMaskedText);

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

                // Update text array with prediction (including the last mask)
                const maskPosition = (i * 2) + 1;
                textArray[maskPosition] = predictions[0].text;
            }

            // After all predictions are done, construct the final complete sentence
            const finalText = textArray.join('');
            console.log('Final completed text:', finalText);

            const inferenceTime = performance.now() - startTime;

            postMessage({
                status: 'complete',
                allPredictions,
                inferenceTime: inferenceTime.toFixed(0),
                completedText: finalText
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