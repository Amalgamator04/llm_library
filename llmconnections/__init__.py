# Expose the main class directly when someone imports the package.
# This allows users to do: 'from llmconnections import LLMConnections'
from .connections import LLMConnections 

# Define the package version, which should match the one in setup.py
__version__ = "1.0.0" 
