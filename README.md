# MALIS Project

This project for our course MALIS implements a binary classification model using gradient descent. It includes data preprocessing, visualization, model training, prediction, and evaluation functions.

## Setup Instructions for Pan

### 1. Clone the Repository

First, clone the repository from GitHub. Replace `<your_username>` with your GitHub username if needed.

```bash
git clone https://github.com/RizPur/MALIS-Project.git
cd MALIS-Project
```

### Set Up a Virtual Environment

It’s recommended to use a virtual environment to manage dependencies.

#### Create virtual environment 

```bash
python3 -m venv venv

```

#### Activate the virtual environment:

- On MacOS/Linux:
    
    ```bash
    source venv/bin/activate
    ```

- On Windows:

    ```ps
    .\venv\Scripts\activate
    ```
You should see (venv) in the terminal prompt, indicating the virtual environment is active.

### Install libraries 

We will use a requirements.txt file to install the necessary dependencies. I put all the libraries we need in this file. So just run:

```bash
pip install -r requirements.txt
```

### Run the project

```bash
python main.py
```

The script will:

1. Load and preprocess the data.
2. Generate visualizations of pairwise feature relationships, saved in plots/binary/pairplot.png.
3. Train the binary classifier using gradient descent.
4. Make predictions on the test set.
5. Evaluate the model and print the accuracy, confusion matrix, and classification report.

### Additional Notes
- Data File: Ensure `iris_binary.csv` is in the root directory, as the code expects this file for data loading.
- Plots Directory: The script saves visualizations to plots/binary/. Make sure the `plots/binary/` directory exists or create it before running the code.
- Deactivating the Virtual Environment: After you’re done, deactivate the virtual environment by running: `deactivate`


### Project Structure

Project Structure

- main.py: Main script for loading data, training the model, and evaluating performance.
- requirements.txt: Dependencies required for the project.
- plots/: Directory where visualizations are saved.


### Summary of Steps
1. **Clone** the repository.
2. **Set up** a virtual environment and **activate** it.
3. **Create `requirements.txt`**, add dependencies, and run `pip install -r requirements.txt`.
4. **Run the script** with `python main.py`.

Good luck Pan!
