# FillMask-JS

## Overview

FillMask-JS is a JavaScript application that demonstrates the **fill-mask** functionality using the [Transformers.js](https://github.com/huggingface/transformers.js) library. Built with React and Vite, this project allows users to input sentences containing mask tokens and receive multiple predictions to fill in the blanks using pre-trained language models.

## Features

- **Model Selection**: Choose from a variety of pre-trained models tailored for different languages and applications.
- **Device Support**: Run inference on WebGPU or WASM backend.
- **Quantization Options**: Optimize performance using various quantization techniques.
- **Real-time Progress Indicators**: Visual feedback during model loading and inference.
- **Responsive UI**: Intuitive and accessible interface built with React.

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

1. **Load a Model**: Select a pre-trained model, device, and quantization type from the dropdowns and click **Load Model**.
2. **Input Text**: Enter a sentence containing a mask token `<mask>` or `..` (double period).
3. **Unmask**: Click the **Unmask** button to get predictions filling in the masked token.

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
