## 🤝 Contributing to DOOM

We welcome contributions to DOOM! This section outlines how you can contribute to the project.

### 🚀 Getting Started

1. **Fork the Repository**
   - Fork the [DOOM repository](https://github.com/Atarilab/DOOM) to your GitHub account
   - Clone your fork locally: `git clone https://github.com/YOUR_USERNAME/DOOM.git`

2. **Set Up Development Environment**
   - Follow the [Installation Instructions](#-installation-instructions) to set up your development environment
   - Ensure all tests pass before making changes

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### 📝 Development Guidelines

#### Code Style
- Follow the existing code formatting using [black](https://github.com/psf/black) and [flake8](https://github.com/PyCQA/flake8)
- Run formatting before committing: `Ctrl + Shift + B` in VS Code and select "Format and Lint"
- Ensure all code follows Python PEP 8 standards

#### Documentation
- Add docstrings to all new functions and classes
- Update relevant documentation files
- Include examples for new features
- Update the README if adding new functionality

#### Testing
- Test your changes in simulation before testing on real hardware
- Add unit tests for new functionality when applicable
- Ensure existing tests continue to pass

### 🔧 Making Changes

#### Adding New Controllers
1. Create a new controller class inheriting from `ControllerBase`
2. Implement the `compute_lowlevelcmd` method
3. Add the controller to the robot's `available_controllers`
4. Update task configurations if needed
5. Add appropriate joystick mappings if required

#### Adding New Robots
1. Create a new robot class inheriting from `RobotBase`
2. Define robot-specific parameters and configurations
3. Add MuJoCo model if simulation support is needed
4. Update task configurations to include the new robot

#### Adding New Tasks
1. Define the task configuration in `task_configs.json`
2. Follow the naming convention: `<controller-type>-<method>-<interface>-<robot>`
3. Update relevant documentation

### 🐛 Bug Reports

When reporting bugs, please include:
- **Environment**: OS, Python version, ROS2 version, Docker version
- **Steps to Reproduce**: Clear, step-by-step instructions
- **Expected vs Actual Behavior**: What you expected vs what happened
- **Error Messages**: Full error logs and stack traces
- **Additional Context**: Any relevant system information

### 💡 Feature Requests

When requesting features, please include:
- **Use Case**: Why this feature is needed
- **Proposed Solution**: How you envision the feature working
- **Alternatives Considered**: Other approaches you've considered
- **Mockups/Examples**: If applicable, provide examples or mockups

### 📤 Submitting Changes

1. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

2. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select the main branch as the base
   - Fill out the PR template with:
     - Description of changes
     - Testing performed
     - Screenshots/videos (if applicable)
     - Related issues

### 🔍 Pull Request Review Process

1. **Automated Checks**: Ensure all CI/CD checks pass
2. **Code Review**: At least one maintainer must approve
4. **Documentation**: All changes must be properly documented

### 🛡️ Safety Guidelines

- **Never test untested code on real robots without proper safety measures**
- **Always start with simulation testing**
- **Use the DAMPING mode for safely stopping sending commands to the robot.**
- **Follow the robot-specific safety guidelines**
- **Test in a controlled environment first**
