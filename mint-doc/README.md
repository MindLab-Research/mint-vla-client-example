# Mind Lab Toolkit (MinT) Documentation

Official documentation for the Mind Lab Toolkit (MinT) - a training API for fine-tuning large language models.

## Overview

MinT enables researchers and developers to fine-tune large language models while abstracting away distributed training complexity. The platform handles GPU infrastructure management while users focus on data, algorithms, and training logic.

## Installation

Install the MinT Python package:

```bash
pip install tinker
```

## Authentication

Contact the Mind Lab team to obtain an API key. Once you have your key, set it as an environment variable:

```bash
export TINKER_API_KEY=your_api_key_here
```

## Documentation

This repository contains the source for the MinT documentation website, built with [Nextra](https://nextra.site/).

### Requirements

- Node.js >= 20.9.0
- npm >= 9.0.0

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

Build the documentation:

```bash
npm run build
```

### Production

Start the production server:

```bash
npm start
```

## Features

- **Training API:** Fine-tune open-weight models (Qwen, Llama, DeepSeek)
- **Vision-Language Support:** Train multimodal models
- **LoRA Fine-tuning:** Efficient adaptation with low-rank methods
- **Multiple Training Paradigms:** Supervised learning, reinforcement learning, preference learning
- **Flexible Loss Functions:** Built-in and custom loss functions
- **Model Management:** Save, load, and publish trained models

## Repository

- **Documentation:** This repository
- **Cookbook:** [github.com/thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)

## License

ISC
