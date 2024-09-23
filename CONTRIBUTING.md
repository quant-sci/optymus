# Contributing to optymus

Thank you for considering contributing to `optymus`! By contributing to this project, you help make it a more powerful and versatile optimization methods library for the Python community.

## How to Contribute

If you're interested in contributing to `optymus`, please follow these guidelines:

### 1. Fork the Repository

Click on the "Fork" button on the top right corner of the [optymus repository](https://github.com/quant-sci/optymus) to create your own fork.

### 2. Clone your Fork

Clone the repository to your local machine using the following command (replace `your-username` with your GitHub username):

```bash
git clone https://github.com/your-username/optymus.git
cd optymus
```

### 3. Create a conda environment (recommended) and install hatch 

```bash
conda create -n "optymus-env"
conda activate optymus-env

pip install hatch
hatch shell
```

### 4. Create a Branch

Create a new branch for your contribution:

```bash
git checkout -b feature/your-contribution
```

### 5. Make Changes

Make your changes and additions to the codebase. Ensure that your code follows the project's coding standards.

### 6. Run Tests

```bash
hatch run testing:run
```

### 7. Commit Changes

Commit your changes with a clear and descriptive commit message:

```bash
git add .
git commit -m "add feature: your contribution"
```

### 8. Push Changes

Push your changes to your forked repository on GitHub:

```bash
git push origin feature/your-contribution
```

### 9. Submit a Pull Request

Open a pull request (PR) on the main `optymus` repository. Provide a detailed description of your changes and why they are valuable. Reference any related issues in your PR description.

## Code Style

Follow the existing code style and conventions used in the project. If there are specific style guidelines, they will be outlined in the project's documentation.

## Testing

If your contribution includes new features or modifications, it is encouraged to include tests. Tests should be located in the `tests` directory and be named appropriately.

## Issue Tracker

Before starting work on a new feature or bug fix, check the [issue tracker](https://github.com/quant-sci/optymus/issues) for existing issues. If you plan to address an existing issue, please comment on it to let others know you are working on it.

## Thank You!

Your contributions help make `optymus` a valuable resource for the Python community. Thank you for your time and effort!