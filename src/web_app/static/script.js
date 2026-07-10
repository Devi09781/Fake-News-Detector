// Fake News Detector Web App - JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const textForm = document.getElementById('text-form');
    const urlForm = document.getElementById('url-form');
    const resultsSection = document.getElementById('results-section');
    const resultCard = document.getElementById('result-card');
    const resultLabel = document.getElementById('result-label');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceLevel = document.getElementById('confidence-level');
    const extractedText = document.getElementById('extracted-text');
    const spinnerContainer = document.getElementById('spinner-container');
    const clearBtn = document.getElementById('clear-btn');
    
    // Event Listeners
    if (textForm) {
        textForm.addEventListener('submit', function(e) {
            e.preventDefault();
            analyzeText();
        });
    }
    
    if (urlForm) {
        urlForm.addEventListener('submit', function(e) {
            e.preventDefault();
            analyzeUrl();
        });
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            clearResults();
        });
    }
    
    // Functions
    function analyzeText() {
        const textInput = document.getElementById('text-input').value.trim();
        if (!textInput) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        const selectedModel = getSelectedModel();
        if (!selectedModel) {
            alert('Please select a model for analysis.');
            return;
        }
        
        // Show loading spinner
        showSpinner();
        
        // Prepare data for the request
        const data = {
            text: textInput,
            model: selectedModel
        };
        
        // Send the request to the server
        fetch('/analyze_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Hide spinner
            hideSpinner();
            
            // Display the results
            displayResults(data);
        })
        .catch(error => {
            hideSpinner();
            console.error('Error:', error);
            alert('An error occurred while analyzing the text. Please try again.');
        });
    }
    
    function analyzeUrl() {
        const urlInput = document.getElementById('url-input').value.trim();
        if (!urlInput) {
            alert('Please enter a URL to analyze.');
            return;
        }
        
        // Validate URL format
        if (!isValidUrl(urlInput)) {
            alert('Please enter a valid URL (e.g., https://example.com).');
            return;
        }
        
        const selectedModel = getSelectedModel();
        if (!selectedModel) {
            alert('Please select a model for analysis.');
            return;
        }
        
        // Show loading spinner
        showSpinner();
        
        // Prepare data for the request
        const data = {
            url: urlInput,
            model: selectedModel
        };
        
        // Send the request to the server
        fetch('/analyze_url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // Hide spinner
            hideSpinner();
            
            // Display the results
            displayResults(data);
        })
        .catch(error => {
            hideSpinner();
            console.error('Error:', error);
            alert('An error occurred while analyzing the URL. Please try again.');
        });
    }
    
    function displayResults(data) {
        // Show results section
        resultsSection.classList.add('active');
        
        // Set result label and class
        resultLabel.textContent = data.prediction;
        
        // Remove previous classes
        resultCard.classList.remove('result-real', 'result-fake', 'result-uncertain');
        
        // Add appropriate class based on prediction
        if (data.prediction === 'REAL') {
            resultCard.classList.add('result-real');
        } else if (data.prediction === 'FAKE') {
            resultCard.classList.add('result-fake');
        } else {
            resultCard.classList.add('result-uncertain');
        }
        
        // Set confidence value and level
        const confidencePercent = (data.confidence * 100).toFixed(2);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceLevel.style.width = `${confidencePercent}%`;
        
        // Set extracted text if available
        if (data.extracted_text) {
            extractedText.textContent = data.extracted_text;
            document.getElementById('extracted-text-container').style.display = 'block';
        } else {
            document.getElementById('extracted-text-container').style.display = 'none';
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    function getSelectedModel() {
        const modelRadios = document.getElementsByName('model');
        for (const radio of modelRadios) {
            if (radio.checked) {
                return radio.value;
            }
        }
        return null;
    }
    
    function isValidUrl(url) {
        try {
            new URL(url);
            return true;
        } catch (error) {
            return false;
        }
    }
    
    function showSpinner() {
        spinnerContainer.style.display = 'flex';
    }
    
    function hideSpinner() {
        spinnerContainer.style.display = 'none';
    }
    
    function clearResults() {
        // Clear input fields
        if (document.getElementById('text-input')) {
            document.getElementById('text-input').value = '';
        }
        
        if (document.getElementById('url-input')) {
            document.getElementById('url-input').value = '';
        }
        
        // Hide results section
        resultsSection.classList.remove('active');
    }
});