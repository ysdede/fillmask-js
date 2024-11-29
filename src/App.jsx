import React, { useState, useEffect } from 'react'
import './App.css'

const AVAILABLE_MODELS = [
    "Xenova/bert-base-uncased",
    "Xenova/bert-base-cased",
    "Xenova/roberta-base",
    "serdarcaglar/roberta-base-turkish-scientific-cased-ONNX",
    "ysdede/roberta-tr-rad-v1-onnx",
    "Xenova/Bio_ClinicalBERT",
    "Xenova/distilbert-base-cased",
    "Xenova/distilbert-base-uncased",
    "Xenova/bert-base-multilingual-uncased",
    "Xenova/albert-base-v2",
    "Xenova/bert-base-multilingual-cased",
    "Xenova/xlm-roberta-base",
    "Xenova/distilroberta-base",
    "Xenova/camembert-base",
    "Xenova/macbert4csc-base-chinese",
    "fkrasnov2/COLD2",
    "Xenova/albert-large-v2",
    "Xenova/bert-base-chinese",
    "Xenova/xlm-mlm-100-1280",
    "Xenova/wangchanberta-base-att-spm-uncased",
    "Xenova/xlm-clm-ende-1024",
    "Xenova/xlm-mlm-ende-1024",
    "Xenova/xlm-mlm-tlm-xnli15-1024",
    "Xenova/xlm-mlm-xnli15-1024",
    "benjaminchazelle/camembert-onnx",
    "Xenova/ernie-3.0-base-zh",
    "Xenova/ernie-3.0-medium-zh",
    "Xenova/xlm-clm-enfr-1024",
    "Xenova/xlm-mlm-17-1280",
    "Xenova/antiberta2",
    "Xenova/ernie-3.0-xbase-zh",
    "Xenova/ernie-1.0-base-zh",
    "Xenova/xlm-mlm-en-2048",
    "r3sgame/bert-base-cased",
    "Xenova/xlm-mlm-enfr-1024",
    "Xenova/xlm-mlm-enro-1024",
];

const DEVICE_TYPES = [
    { value: 'webgpu', label: 'WebGPU' },
    { value: 'wasm', label: 'WASM' },
];

const QUANTIZATION_TYPES = [
    { value: 'fp32', label: 'FP32 (32-bit float)' },
    { value: 'fp16', label: 'FP16 (16-bit float)' },
    { value: 'q8', label: 'Q8 (8-bit quantization)' },
    { value: 'int8', label: 'INT8 (8-bit integer quantization)' },
    { value: 'uint8', label: 'UINT8 (8-bit unsigned integer quantization)' },
    { value: 'q4', label: 'Q4 (4-bit quantization)' },
    { value: 'q4f16', label: 'Q4F16 (4-bit quantization with 16-bit float precision)' },
    { value: 'bnb4', label: 'BNB4 (4-bit Bitsandbytes quantization)' },
];

const SPARE_PLACEHOLDERS = [
    { value: 'model_mask', label: 'Model Mask Token' },
    { value: '..', label: 'Double Period (..)' },
    { value: ' ', label: 'Single Space ( )' },
    { value: 'custom', label: 'Custom...' }
];

function App() {
    const [inputText, setInputText] = useState('I had come there not ?? to look at, but ?? to number myself sincerely and wholeheartedly with, the mob. As for my secret moral views, I ?? no room for them amongst my actual, practical ??.');
    const [predictions, setPredictions] = useState([]);
    const [status, setStatus] = useState('idle');
    const [message, setMessage] = useState('');
    const [worker, setWorker] = useState(null);
    const [selectedModel, setSelectedModel] = useState('Xenova/bert-base-cased');
    const [isModelLoaded, setIsModelLoaded] = useState(false);
    const [loadTime, setLoadTime] = useState(null);
    const [inferenceTime, setInferenceTime] = useState(null);
    const [selectedDevice, setSelectedDevice] = useState('wasm');
    const [selectedQuantization, setSelectedQuantization] = useState('q8');
    const [notifications, setNotifications] = useState([]);
    const notificationIdRef = React.useRef(0);
    const [canLoadModel, setCanLoadModel] = useState(true);
    const [currentConfig, setCurrentConfig] = useState(null);
    const [useSequentialUnmasking, setUseSequentialUnmasking] = useState(true);
    const [sparePlaceholder, setSparePlaceholder] = useState('model_mask');
    const [customPlaceholder, setCustomPlaceholder] = useState('');
    const [modelMaskToken, setModelMaskToken] = useState('[MASK]');
    const [completedSentence, setCompletedSentence] = useState('');

    useEffect(() => {
        document.title = 'transformers.js fill-mask demo';
        
        return () => {
            if (worker) {
                worker.terminate();
            }
        };
    }, []);

    const addNotification = (message, type = 'info', duration = 5000) => {
        const id = ++notificationIdRef.current;
        setNotifications(prev => [...prev, { id, message, type }]);
        
        setTimeout(() => {
            setNotifications(prev => prev.filter(n => n.id !== id));
        }, duration);
    };

    const createWorker = () => {
        const mlmWorker = new Worker(new URL('./worker_mlm.js', import.meta.url), { type: 'module' });

        mlmWorker.onmessage = (event) => {
            const { status, predictions, allPredictions, message, loadTime, inferenceTime, duration, maskToken } = event.data;

            if (status === 'loading' || status === 'progress') {
                setStatus('loading');
                setMessage(message);
            } else if (status === 'complete') {
                setStatus('complete');
                setPredictions(allPredictions);
                setMessage('');
                setInferenceTime(inferenceTime);
                setCompletedSentence(event.data.completedText);
            } else if (status === 'error') {
                setStatus('error');
                setPredictions(predictions);
                setMessage('');
                addNotification(predictions[0], 'error', 6000);
            } else if (status === 'model_loaded') {
                setIsModelLoaded(true);
                setStatus('idle');
                setMessage('');
                setLoadTime(loadTime);
                setModelMaskToken(maskToken);
            } else if (status === 'info' || status === 'warning') {
                addNotification(message, status, duration || 5000);
            }
        };

        return mlmWorker;
    };

    const getConfigString = (model, device, quantization) => {
        return `${model}-${device}-${quantization}`;
    };

    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
        const newConfig = getConfigString(e.target.value, selectedDevice, selectedQuantization);
        setCanLoadModel(newConfig !== currentConfig);
    };

    const handleDeviceChange = (e) => {
        setSelectedDevice(e.target.value);
        const newConfig = getConfigString(selectedModel, e.target.value, selectedQuantization);
        setCanLoadModel(newConfig !== currentConfig);
    };

    const handleQuantizationChange = (e) => {
        setSelectedQuantization(e.target.value);
        const newConfig = getConfigString(selectedModel, selectedDevice, e.target.value);
        setCanLoadModel(newConfig !== currentConfig);
    };

    const handleLoadModel = () => {
        const configString = getConfigString(selectedModel, selectedDevice, selectedQuantization);
        setCurrentConfig(configString);
        setCanLoadModel(false);

        if (worker) {
            worker.terminate();
        }

        const newWorker = createWorker();
        setWorker(newWorker);

        setIsModelLoaded(false);
        setStatus('loading');
        setMessage('Initializing model...');
        setLoadTime(null);
        setPredictions([]);
        setInferenceTime(null);

        setTimeout(() => {
            newWorker.postMessage({ 
                type: 'load_model', 
                model: selectedModel,
                device: selectedDevice,
                quantization: selectedQuantization
            });
        }, 100);

        // Disable controls after loading
        setSelectedModel(prevModel => {
            document.querySelector('.model-selector select').disabled = true;
            document.querySelector('.device-select').disabled = true;
            document.querySelector('.quantization-select').disabled = true;
            return prevModel;
        });
    };

    const handlePredict = () => {
        if (worker && isModelLoaded) {
            setStatus('loading');
            setMessage('Processing...');
            setInferenceTime(null);
            worker.postMessage({ 
                type: 'predict', 
                text: inputText,
                useSequential: useSequentialUnmasking,
                sparePlaceholder: sparePlaceholder === 'model_mask' ? modelMaskToken : 
                                sparePlaceholder === 'custom' ? customPlaceholder : 
                                sparePlaceholder
            });
        }
    };

    return (
        <div className="container">
            <h2>transformers.js fill-mask demo</h2>
            <div className="card">
                <div className="input-section">
                    <textarea
                        rows="2"
                        cols="50"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        placeholder="Enter a sentence with <mask> or ?? (double question mark)"
                    />
                    
                    <div className="input-footer">
                        <div className="usage-hint">
                            You can type '??' (double question mark) instead of '&lt;mask&gt;' for faster input
                        </div>
                        <button
                            onClick={handlePredict}
                            disabled={status === 'loading' || !isModelLoaded}
                            className="unmask-button"
                        >
                            {status === 'loading' ? 'Processing...' : 'Unmask'}
                        </button>
                    </div>
                </div>

                <div className="completed-sentence">
                    {completedSentence && status === 'complete' && (
                        <>
                            <span className="completed-label">Completed sentence:</span>
                            <span className="completed-text">{completedSentence}</span>
                        </>
                    )}
                </div>

                <div className="sequential-option">
                    <label>
                        <input
                            type="checkbox"
                            checked={useSequentialUnmasking}
                            onChange={(e) => setUseSequentialUnmasking(e.target.checked)}
                        />
                        Use previous predictions for sequential unmasking
                    </label>
                </div>

                <div className="spare-placeholder-option">
                    <label>Placeholder for unpredicted masks:</label>
                    <select 
                        value={sparePlaceholder} 
                        onChange={(e) => {
                            setSparePlaceholder(e.target.value);
                            if (e.target.value !== 'custom') {
                                setCustomPlaceholder('');
                            }
                        }}
                    >
                        {SPARE_PLACEHOLDERS.map(option => (
                            <option key={option.value} value={option.value}>
                                {option.value === 'model_mask' 
                                    ? `Model Mask Token (${modelMaskToken})` 
                                    : option.label}
                            </option>
                        ))}
                    </select>
                    {sparePlaceholder === 'custom' && (
                        <input
                            type="text"
                            value={customPlaceholder}
                            onChange={(e) => setCustomPlaceholder(e.target.value)}
                            placeholder="Enter custom placeholder"
                            className="custom-placeholder-input"
                        />
                    )}
                </div>

                {status === 'complete' && predictions.length > 0 && (
                    <div className="all-predictions-container">
                        {predictions.map((predictionSet, setIndex) => (
                            <div key={setIndex} className="predictions-container">
                                <div className="mask-label">
                                    Mask #{predictionSet.maskIndex} 
                                    <span className="mask-info">
                                        (original: {predictionSet.originalPattern})
                                    </span>
                                </div>
                                <div className="inference-text">
                                    Raw text: {predictionSet.inferenceText}
                                </div>
                                <div className="predictions-scroll horizontal">
                                    {predictionSet.predictions.map((pred, index) => (
                                        <div key={index} className="prediction-item">
                                            {pred}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {inferenceTime && (
                    <div className="benchmark-info">
                        Inference time: {inferenceTime}ms
                    </div>
                )}

                <div className="model-section">
                    <h3>Model Selection</h3>
                    <div className="model-selector">
                        <select
                            value={selectedModel}
                            onChange={handleModelChange}
                            disabled={status === 'loading'}
                        >
                            {AVAILABLE_MODELS.map(model => (
                                <option key={model} value={model}>{model}</option>
                            ))}
                        </select>
                        <select
                            value={selectedDevice}
                            onChange={handleDeviceChange}
                            disabled={status === 'loading'}
                            className="device-select"
                        >
                            {DEVICE_TYPES.map(device => (
                                <option key={device.value} value={device.value}>
                                    {device.label}
                                </option>
                            ))}
                        </select>
                        <select
                            value={selectedQuantization}
                            onChange={handleQuantizationChange}
                            disabled={status === 'loading'}
                            className="quantization-select"
                        >
                            {QUANTIZATION_TYPES.map(quant => (
                                <option key={quant.value} value={quant.value}>
                                    {quant.label}
                                </option>
                            ))}
                        </select>
                        <button
                            onClick={handleLoadModel}
                            disabled={status === 'loading' || !canLoadModel}
                            className="load-button"
                        >
                            {status === 'loading' ? 'Loading...' : 'Load Model'}
                        </button>
                    </div>

                    {loadTime && (
                        <div className="benchmark-info">
                            Model load time: {loadTime}ms ({selectedDevice})
                        </div>
                    )}
                </div>

                {message && (
                    <div className="message">
                        {message}
                    </div>
                )}

                {status === 'error' && predictions.length > 0 && (
                    <div className="error">
                        {predictions[0]}
                    </div>
                )}
            </div>

            <div className="notifications-container">
                {notifications.map(({ id, message, type }) => (
                    <div key={id} className={`notification notification-${type}`}>
                        {message}
                    </div>
                ))}
            </div>
        </div>
    )
}

export default App
