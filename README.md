# FillMask-JS

## Overview

A web-based masked language model demo using [Transformers.js](https://github.com/huggingface/transformers.js) library. Built with React and Vite.

**Alternatively, you can use the application directly on Huggingface Spaces without any installation:** [huggingface.co/spaces/ysdede/fill-mask-demo](https://huggingface.co/spaces/ysdede/fill-mask-demo)

## Features

- Support for multiple BERT/RoBERTa models
- WebGPU and WASM backend support
- Multiple quantization options
- Sequential token prediction
  - Predict multiple masked tokens sequentially using previous predictions
  - Option to toggle between sequential and parallel prediction modes
- Configurable mask placeholders
  - Use model's mask token ([MASK])
  - Double period (..)
  - Single space
  - Custom placeholder text
- Real-time performance metrics
  - Model load time
  - Inference time per prediction
- Comprehensive prediction results
  - Shows completed sentence
  - Displays original mask pattern
  - Shows inference text for each mask
  - Multiple token predictions per mask

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/ysdede/fillmask-js.git
    cd fillmask-js
    ```

2. **Install dependencies**:

    ```bash
    npm install
    ```

3. **Run the development server**:

    ```bash
    npm run dev
    ```

4. **Build for production**:

    ```bash
    npm run build
    ```

5. **Preview the production build**:

    ```bash
    npm run preview
    ```

## Usage

1. Select a model, backend (WebGPU/WASM) and quantization level
2. Load the model
3. Enter text with masks (use ?? or <mask> tokens)
4. Choose prediction mode:
   - Sequential: Uses previous predictions for subsequent masks
   - Parallel: Predicts all masks independently
5. Select placeholder type for unpredicted masks
6. Click "Unmask" to get predictions

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

1. **Fork the repository**
2. **Create a new branch**: `git checkout -b feature/YourFeature`
3. **Commit your changes**: `git commit -m 'Add some feature'`
4. **Push to the branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Transformers.js](https://github.com/huggingface/transformers.js)
- [React](https://reactjs.org/)
- [Vite](https://vitejs.dev/)
- [Hugging Face](https://huggingface.co/)
