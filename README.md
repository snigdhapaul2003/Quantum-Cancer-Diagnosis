# Quantum Cancer Diagnosis (QCD)

![QCD Logo](static/img/logo.png)

## Unlocking Tomorrow's Cures: Quantum-Powered Cancer Diagnosis

QCD is an advanced healthcare platform that combines quantum computing techniques with artificial intelligence to provide accurate breast cancer diagnosis through histopathological image analysis.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Quantum Advantage](#quantum-advantage)
- [Installation](#installation)
- [Usage](#usage)
- [Multilingual Support](#multilingual-support)
- [Medical Professional Integration](#medical-professional-integration)
- [Demo Videos](#demo-videos)
- [Contributing](#contributing)
- [Disclaimers](#disclaimers)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

### Core Features
- **Quantum-Classical Hybrid Stack**: Leverages both quantum and classical algorithms for superior diagnostic accuracy
- **Histopathological Image Analysis**: Analyzes microscopic images of breast tissue to detect cancer
- **Automated Report Generation**: Creates detailed PDF reports of diagnosis results
- **WhatsApp Integration**: Sends reports and notifications via WhatsApp for ease of access
- **Email Support**: Delivers reports to patients via email
- **AI-Powered Assistant**: Conversational AI to answer patient questions about their diagnosis

### Classification Categories
- **Benign Categories**:
  - Adenosis
  - Fibroadenoma
  - Phyllodes tumor
  - Tubular adenoma
  
- **Malignant Categories**:
  - Ductal carcinoma
  - Lobular carcinoma
  - Mucinous carcinoma
  - Papillary carcinoma

### Added Functionalities
- **Doctor Recommendation System**: Suggests appropriate specialists based on diagnosis, location, and availability
- **Multilingual Interface**: Available in English, Hindi, and Bengali
- **Follow-up Support**: AI assistant for post-diagnosis support and questions

## Technology Stack

### Backend
- **Python**: Core programming language
- **Flask**: Web framework for the application
- **TensorFlow/Keras**: Deep learning framework for image analysis
- **Quantum Computing Libraries**: Integration with quantum computing capabilities
- **YOLOv8**: For object detection in medical images
- **PostgreSQL**: Database for storing patient records and diagnoses

### AI/ML
- **DenseNet121**: Pre-trained model for feature extraction
- **Support Vector Classifiers**: Both classical and quantum variants
- **Nearest Neighbors**: For doctor recommendation system
- **NSGA-II Algorithm**: Multi-objective optimization for doctor recommendations

### Frontend
- **HTML/CSS**: Structure and styling
- **JavaScript**: Interactive elements
- **Bootstrap**: Responsive design framework

### APIs and Services
- **Twilio**: WhatsApp integration for patient communication
- **Gmail API**: Email delivery of reports
- **Groq API**: AI language model for medical conversations
- **Google Gemini API**: For advanced image analysis verification
- **Geolocation Services**: For doctor location recommendations

## Architecture

QCD employs a multi-layered architecture:

1. **Input Layer**: Accepts histopathological images and patient information
2. **Image Processing Layer**: Preprocesses images for analysis
3. **Feature Extraction Layer**: Uses DenseNet121 for extracting key image features
4. **Quantum-Classical Hybrid Layer**: Combines quantum and classical algorithms for diagnosis
5. **Classification Layer**: Determines benign/malignant status and specific cancer type
6. **Report Generation Layer**: Creates and distributes diagnosis reports
7. **Communication Layer**: Handles patient interactions via WhatsApp and email
8. **Doctor Recommendation Layer**: Suggests appropriate specialists based on diagnosis

## Quantum Advantage

QCD leverages quantum computing techniques to enhance traditional machine learning approaches:

- **Enhanced Feature Processing**: Quantum algorithms process complex feature correlations
- **Improved Classification Accuracy**: Quantum Support Vector Machines complement classical models
- **Hybrid Approach**: Combines the strengths of quantum and classical computing for optimal results
- **Better Pattern Recognition**: Quantum algorithms detect subtle patterns that classical methods might miss

## Installation

### Prerequisites
- Python 3.7+
- PostgreSQL
- Required Python packages (see requirements.txt)
- Twilio account for WhatsApp integration
- Gmail API credentials
- Quantum computing environment (optional for full functionality)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/snigdhapaul2003/Quantum-Cancer-Diagnosis.git
cd Quantum-Cancer-Diagnosis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your API keys and credentials
```

5. Set up the PostgreSQL database:
```bash
# Create a database named 'QCD'
# Update database credentials in .env file
```

6. Start the application:
```bash
cd src
python app.py
```

## Usage

### Patient Diagnosis Flow

1. **Access the Platform**: Visit the web interface or access via mobile device
2. **Input Information**: Provide basic details (name, age, contact)
3. **Upload Image**: Submit histopathological image for analysis
4. **Review Results**: View diagnosis results on the platform
5. **Receive Report**: Get detailed report via WhatsApp and email
6. **Connect with Specialists**: Use doctor recommendation system to find appropriate healthcare providers
7. **Follow Up**: Use the AI assistant for additional questions about the diagnosis

### Healthcare Provider Integration

1. **Register as a Provider**: Add your details to the database
2. **Receive Referrals**: Get connected with patients based on diagnosis needs
3. **View Patient Reports**: Access comprehensive diagnosis reports

## Multilingual Support

QCD is available in multiple languages to serve diverse populations:

- English (Default)
- Hindi (हिंदी)
- Bengali (বাংলা)

Language can be selected from the main interface navigation menu.

## Medical Professional Integration

QCD includes features for medical professionals:

- **Doctor Database**: Registered doctors organized by speciality, location and availability
- **Appointment System**: Helps patients book appointments with appropriate specialists
- **Specialist Matching**: Uses multi-objective optimization to match patients with doctors based on:
  - Location proximity
  - Specialization relevance
  - Experience level
  - Rating
  - Availability

## Demo Videos

- [System Overview](https://youtu.be/5GS2NOJxaFs)
- [Using the Detection System](https://youtu.be/V4VAc0rfeps)

## Contributing

We welcome contributions to enhance QCD's capabilities:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Disclaimers

This application is provided for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

The AI-generated reports are preliminary and should be reviewed by qualified healthcare professionals before making medical decisions. The system is designed to assist medical professionals, not replace them.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The development team at QCD
- Medical advisors and healthcare partners
- Open-source contributors and libraries
- Testing and feedback providers
