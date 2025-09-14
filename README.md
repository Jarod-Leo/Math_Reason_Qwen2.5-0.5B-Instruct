# GRPO Training Script

This project implements a training script for the GRPO (Generalized Reinforcement Policy Optimization) model using the Qwen2.5-0.5B-Instruct model. The script is designed to preprocess a dataset, configure training parameters, and initiate the training process.

## Project Structure

- `main.py`: Entry point for the application. Contains the main logic for loading the model, preprocessing the dataset, configuring training parameters, and starting the training process.
- `requirements.txt`: Lists the dependencies required for the project.
- `config.py`: Contains configuration settings for the training process, including model parameters and training arguments.
- `utils.py`: Includes utility functions for data preprocessing, extracting answers from model outputs, and defining reward functions.

## Installation

To set up the environment, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd grpo-training-script
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the training script, execute the following command:

```
python main.py
```

Make sure to adjust any configuration settings in `config.py` as needed before running the script.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
