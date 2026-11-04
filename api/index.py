import sys
import os

# Add parent directory to sys.path so app can be imported by Vercel's serverless handler
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
