import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the Fake News Detector web application')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the application on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the application on')
    parser.add_argument('--debug', action='store_true', help='Run the application in debug mode')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing the trained models')
    
    args = parser.parse_args()
    
    # Add the src directory to the path
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.append(src_dir)
    
    # Import the Flask app
    from web_app.app import app
    
    # Set the models directory
    app.config['MODELS_DIR'] = os.path.abspath(args.models_dir)
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()