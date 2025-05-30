#!/usr/bin/env python3
"""
Patient Database Chat GUI Application

A PyQt6 graphical interface chat application that uses the Agno framework
to interact with the patient database via natural language.

Author: Hao Jiang, UTSW Medical Center
"""

import os
import sys
import json
import sqlite3
import datetime
import uuid
import asyncio
import threading
import re
import time
import markdown
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QFrame,
    QScrollArea, QMessageBox, QProgressBar, QStatusBar, QMenuBar,
    QMenu, QToolBar, QSizePolicy, QTextBrowser, QGroupBox, QAbstractScrollArea,
    QSlider, QStackedWidget
)
# OpenGL widgets removed - using software-based 3D viewer instead
# from PyQt6.QtOpenGLWidgets import QOpenGLWidget
# from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture, QOpenGLVertexArrayObject
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtCore import (
    Qt, QThread, QObject, pyqtSignal, QTimer, QSize, QSettings
)
from PyQt6.QtGui import (
    QFont, QTextCursor, QTextCharFormat, QColor, QPalette, QAction, QIcon, QPixmap
)

from agno.agent import Agent
from agno.storage.sqlite import SqliteStorage
from agno.models.google import Gemini

# OpenGL and 3D visualization imports
try:
    from OpenGL.GL import *
    from OpenGL.GL.shaders import compileProgram, compileShader
    import numpy as np
    import ctypes
    OPENGL_AVAILABLE = True
except ImportError:
    print("Warning: OpenGL not available. 3D viewer will be disabled.")
    OPENGL_AVAILABLE = False
from agno.tools import tool
from agno.tools.thinking import ThinkingTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Constants
DB_FILE = "patient_records.db"
SCHEMA_FILE = "Schema.MD"
AGENT_STORAGE = "tmp/patient_db_agents.db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "Use_Your_API_KEY")

# Ensure tmp directory exists
os.makedirs(os.path.dirname(AGENT_STORAGE), exist_ok=True)

# Markdown Renderer for Chat Messages
class MarkdownRenderer:
    """Render markdown text with syntax highlighting."""
    
    def __init__(self):
        self.formatter = HtmlFormatter(
            style='github-dark',
            noclasses=True,
            cssclass='highlight'
        )
        
    def render_markdown(self, text: str) -> str:
        """Convert markdown text to HTML with syntax highlighting."""
        try:
            # Clean the text first
            text = self._clean_text(text)
            
            # Convert markdown to HTML with improved extensions
            html = markdown.markdown(
                text,
                extensions=[
                    'markdown.extensions.fenced_code',
                    'markdown.extensions.tables',
                    'markdown.extensions.nl2br',
                    'markdown.extensions.codehilite'
                ],
                extension_configs={
                    'markdown.extensions.codehilite': {
                        'css_class': 'codehilite',
                        'use_pygments': True,
                        'pygments_style': 'github-dark',
                        'noclasses': True
                    },
                    'markdown.extensions.fenced_code': {
                        'lang_prefix': 'language-'
                    }
                }
            )
            
            # Apply dark theme styles
            html = self._apply_dark_theme(html)
            
            return html
            
        except Exception as e:
            print(f"Markdown rendering error: {e}")
            # Enhanced fallback with basic formatting
            return self._fallback_formatting(text)
    
    def _process_code_blocks(self, text: str) -> str:
        """Process code blocks with syntax highlighting."""
        # Pattern for fenced code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        
        def replace_code_block(match):
            language = match.group(1) or 'text'
            code = match.group(2)
            
            try:
                if language.lower() in ['sql', 'python', 'json', 'yaml', 'bash', 'shell']:
                    lexer = get_lexer_by_name(language.lower())
                else:
                    lexer = guess_lexer(code)
                
                highlighted = highlight(code, lexer, self.formatter)
                return highlighted
            except:
                # Fallback to plain code block
                return f'<pre style="background-color: #2a2a2a; color: #ffffff; padding: 10px; border-radius: 6px; border: 1px solid #555555;"><code>{code}</code></pre>'
        
        return re.sub(pattern, replace_code_block, text, flags=re.DOTALL)
    
    def _apply_dark_theme(self, html: str) -> str:
        """Apply dark theme styles to HTML."""
        # Modern dark theme with enhanced styling
        dark_css = """
        <style>
        body { 
            color: #e2e8f0; 
            background-color: #1a202c; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.7;
            font-size: 14px;
            font-weight: 400;
        }
        
        /* Modern headings with better hierarchy */
        h1, h2, h3, h4, h5, h6 { 
            color: #63b3ed; 
            margin: 20px 0 12px 0; 
            font-weight: 700;
            letter-spacing: -0.025em;
        }
        h1 { font-size: 28px; color: #90cdf4; border-bottom: 2px solid #4299e1; padding-bottom: 8px; }
        h2 { font-size: 24px; color: #63b3ed; border-bottom: 1px solid #3182ce; padding-bottom: 6px; }
        h3 { font-size: 20px; color: #4299e1; }
        h4 { font-size: 18px; color: #4299e1; }
        h5 { font-size: 16px; color: #4299e1; }
        h6 { font-size: 14px; color: #4299e1; text-transform: uppercase; letter-spacing: 0.05em; }
        
        /* Enhanced paragraph styling */
        p { 
            margin: 12px 0; 
            line-height: 1.7;
            color: #e2e8f0;
        }
        
        /* Better emphasis and strong text */
        strong, b { 
            color: #ffd369; 
            font-weight: 700; 
        }
        em, i { 
            color: #d6bcf0; 
            font-style: italic; 
        }
        
        /* Modern inline code styling */
        code { 
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: #63b3ed; 
            padding: 3px 8px; 
            border-radius: 6px; 
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', 'Source Code Pro', monospace;
            font-size: 13px; 
            border: 1px solid #4a5568;
            font-weight: 500;
        }
        
        /* Enhanced code blocks */
        pre { 
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            border: 2px solid #4a5568; 
            border-radius: 10px; 
            padding: 16px; 
            overflow-x: auto; 
            margin: 16px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            position: relative;
        }
        
        /* Modern blockquotes */
        blockquote { 
            border-left: 4px solid #4299e1; 
            background: linear-gradient(90deg, rgba(66, 153, 225, 0.1) 0%, transparent 100%);
            padding: 16px 20px; 
            margin: 16px 0; 
            border-radius: 0 8px 8px 0;
            color: #cbd5e0; 
            font-style: italic;
            position: relative;
        }
        blockquote:before {
            content: '"';
            color: #4299e1;
            font-size: 48px;
            position: absolute;
            left: 8px;
            top: -8px;
            opacity: 0.3;
        }
        
        /* Enhanced tables */
        table { 
            border-collapse: separate; 
            border-spacing: 0;
            width: 100%; 
            margin: 16px 0; 
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        th, td { 
            border: none;
            padding: 12px 16px; 
            text-align: left; 
            border-bottom: 1px solid #4a5568;
        }
        th { 
            background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
            color: #f7fafc; 
            font-weight: 700; 
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.05em;
        }
        tr:nth-child(even) { 
            background: rgba(45, 55, 72, 0.3); 
        }
        tr:hover {
            background: rgba(66, 153, 225, 0.1);
            transition: background-color 0.2s ease;
        }
        
        /* Better lists */
        ul, ol { 
            margin: 12px 0; 
            padding-left: 24px; 
        }
        li { 
            margin: 6px 0; 
            line-height: 1.6;
        }
        ul li::marker {
            color: #4299e1;
        }
        ol li::marker {
            color: #4299e1;
            font-weight: 700;
        }
        
        /* Horizontal rules */
        hr { 
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #4a5568, transparent);
            margin: 24px 0; 
        }
        
        /* Enhanced syntax highlighting */
        .codehilite {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
            border: 2px solid #4a5568 !important;
            border-radius: 10px !important;
            padding: 16px !important;
            margin: 16px 0 !important;
            overflow-x: auto !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
        }
        .codehilite pre {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .highlight {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
        }
        
        /* Links */
        a {
            color: #63b3ed;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: all 0.2s ease;
        }
        a:hover {
            color: #90cdf4;
            border-bottom-color: #90cdf4;
        }
        
        /* Selection styling */
        ::selection {
            background: rgba(99, 179, 237, 0.3);
        }
        
        /* Scrollbar styling for webkit browsers */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #2d3748;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #4a5568;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #718096;
        }
        </style>
        """
        
        return f"{dark_css}<div>{html}</div>"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better markdown processing."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Fix code block formatting
        text = re.sub(r'```(\w*)\s*\n', r'```\1\n', text)
        
        # Ensure proper line endings for lists
        text = re.sub(r'(?<!\n)\n([-*+]\s)', r'\n\n\1', text)
        
        return text.strip()
    
    def _fallback_formatting(self, text: str) -> str:
        """Provide fallback formatting when markdown fails."""
        # Basic HTML escaping
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Convert basic markdown patterns
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
        
        # Convert newlines
        text = text.replace('\n', '<br>')
        
        return f"<div style='color: #ffffff; background-color: #1e1e1e; padding: 10px;'>{text}</div>"

# Import all the tools from the original console app
@tool(show_result=True)
def load_patient_image(patient_identifier: str = "image", image_type: str = "brain"):
    """Load a patient's medical image (NIfTI format) for display.
    
    This tool loads medical images from the images directory for display in the viewer.
    Use this tool when users ask to view, show, display, or load patient images.
    
    Args:
        patient_identifier (str): Patient ID or identifier (defaults to "image" for generic)
        image_type (str): Type of image (brain, ct, mri, etc.)
        
    Returns:
        dict: Contains status, message, file_path, and image information
    """
    import nibabel as nib
    import os
    
    try:
        # Look for image files in the images directory
        images_dir = "images"
        if not os.path.exists(images_dir):
            return {
                "status": "error",
                "message": f"Images directory '{images_dir}' does not exist."
            }
        
        # Try different filename patterns
        possible_files = [
            f"{patient_identifier}_{image_type}.nii.gz",
            f"patient_{patient_identifier}_{image_type}.nii.gz", 
            f"{patient_identifier}.nii.gz",
            "image.nii.gz"  # default large image (256x256x192)
        ]
        
        image_file = None
        for filename in possible_files:
            filepath = os.path.join(images_dir, filename)
            if os.path.exists(filepath):
                image_file = filepath
                break
        
        if not image_file:
            available_files = os.listdir(images_dir)
            nifti_files = [f for f in available_files if f.endswith('.nii.gz')]
            return {
                "status": "error",
                "message": f"No suitable image found for patient '{patient_identifier}' with type '{image_type}'. Available NIfTI files: {nifti_files}"
            }
        
        # Load the NIfTI image to validate it
        img = nib.load(image_file)
        data = img.get_fdata()
        
        # Get image dimensions
        shape = data.shape
        
        return {
            "status": "success",
            "message": f"Successfully loaded medical image for patient '{patient_identifier}'",
            "file_path": image_file,
            "image_shape": shape,
            "image_type": image_type,
            "num_slices": shape[2] if len(shape) >= 3 else 1,
            "display_info": f"Image loaded: {os.path.basename(image_file)} - Shape: {shape}"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error loading patient image: {str(e)}"
        }

@tool(show_result=True)
def load_segmentation_mask(mask_identifier: str = "mask", mask_type: str = "tumor"):
    """Load a segmentation mask for overlay on the current medical image.
    
    This tool loads segmentation masks (e.g., brain tumors, tissue types) from the images directory
    for overlay display on top of the current medical image. Use this tool when users ask for
    segmentation, tumor detection, brain metastases analysis, or tissue segmentation.
    
    Args:
        mask_identifier (str): Mask file identifier (defaults to "mask" for generic)
        mask_type (str): Type of segmentation (tumor, tissue, brain, mets, etc.)
        
    Returns:
        dict: Contains status, message, file_path, and mask information
    """
    import nibabel as nib
    import os
    import numpy as np
    
    try:
        # Look for mask files in the images directory
        images_dir = "images"
        if not os.path.exists(images_dir):
            return {
                "status": "error",
                "message": f"Images directory '{images_dir}' does not exist."
            }
        
        # Try different filename patterns for masks
        possible_files = [
            f"{mask_identifier}_{mask_type}.nii.gz",
            f"{mask_identifier}.nii.gz",
            f"brain_{mask_type}_mask.nii.gz",
            "targetmaskimage.nii.gz",  # default target mask
            "mask.nii.gz",  # fallback generic mask
            "brain_tissue_mask.nii.gz"  # tissue segmentation fallback
        ]
        
        mask_file = None
        for filename in possible_files:
            filepath = os.path.join(images_dir, filename)
            if os.path.exists(filepath):
                mask_file = filepath
                break
        
        if not mask_file:
            available_files = os.listdir(images_dir)
            mask_files = [f for f in available_files if 'mask' in f.lower() and f.endswith('.nii.gz')]
            return {
                "status": "error",
                "message": f"No suitable mask found for '{mask_identifier}' with type '{mask_type}'. Available mask files: {mask_files}"
            }
        
        # Load the mask to validate it
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        
        # Get mask dimensions and label information
        shape = mask_data.shape
        unique_labels = np.unique(mask_data)
        num_labels = len(unique_labels) - 1  # Exclude background (0)
        
        # Actually load the mask into the NIfTI viewer if it exists
        try:
            # Get the GUI instance and load the mask
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                # Find the main window
                for widget in app.topLevelWidgets():
                    if hasattr(widget, 'nifti_viewer') and widget.nifti_viewer:
                        success, load_message = widget.nifti_viewer.load_mask(mask_file)
                        if success:
                            print(f"DEBUG: Mask loaded into viewer: {load_message}")
                        else:
                            print(f"DEBUG: Failed to load mask into viewer: {load_message}")
                        break
        except Exception as e:
            print(f"DEBUG: Error loading mask into viewer: {e}")
        
        return {
            "status": "success",
            "message": f"Successfully loaded segmentation mask for '{mask_identifier}' ({mask_type})",
            "file_path": mask_file,
            "mask_shape": shape,
            "mask_type": mask_type,
            "num_slices": shape[2] if len(shape) >= 3 else 1,
            "unique_labels": unique_labels.tolist(),
            "num_labels": num_labels,
            "display_info": f"Mask loaded: {os.path.basename(mask_file)} - Shape: {shape}, Labels: {num_labels}"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error loading segmentation mask: {str(e)}"
        }

@tool(show_result=True)
def submit_segmentation_task(patient_id: str, image_file_path: str = None, segmentation_type: str = "OAR"):
    """Submit a segmentation task for external processing.
    
    Creates a task file for external segmentation services to process. The task file contains
    the MR image file path and segmentation type. External services will generate corresponding
    segmentation mask files in the images folder.
    
    Args:
        patient_id (str): Patient identifier for the task
        image_file_path (str): Path to the MR image file (auto-detects if not provided)
        segmentation_type (str): Type of segmentation - "OAR" (Organs at Risk) or "BM" (Brain Metastases)
        
    Returns:
        dict: Status and task file information
    """
    import json
    import os
    import datetime
    
    try:
        # Validate segmentation type
        valid_types = ["OAR", "BM"]
        if segmentation_type.upper() not in valid_types:
            return {
                "status": "error",
                "message": f"Invalid segmentation type '{segmentation_type}'. Must be one of: {valid_types}"
            }
        
        segmentation_type = segmentation_type.upper()
        
        # Auto-detect image file if not provided
        if not image_file_path:
            images_dir = "images"
            if not os.path.exists(images_dir):
                return {
                    "status": "error",
                    "message": f"Images directory '{images_dir}' does not exist."
                }
            
            # Look for common image file patterns
            possible_files = [
                f"{patient_id}.nii.gz",
                f"{patient_id}_image.nii.gz",
                f"{patient_id}_mr.nii.gz",
                "image.nii.gz",  # Default image file
                "brain_image.nii.gz"
            ]
            
            image_file = None
            for filename in possible_files:
                filepath = os.path.join(images_dir, filename)
                if os.path.exists(filepath):
                    image_file = filepath
                    break
            
            if not image_file:
                available_files = os.listdir(images_dir)
                nifti_files = [f for f in available_files if f.endswith('.nii.gz')]
                return {
                    "status": "error",
                    "message": f"No suitable image found for patient '{patient_id}'. Available NIfTI files: {nifti_files}"
                }
            
            image_file_path = image_file
        else:
            # Validate provided file path
            if not os.path.exists(image_file_path):
                return {
                    "status": "error",
                    "message": f"Image file not found: {image_file_path}"
                }
        
        # Ensure tasks directory exists
        tasks_dir = "tasks"
        os.makedirs(tasks_dir, exist_ok=True)
        
        # Create task file content
        task_data = {
            "task_id": f"{patient_id}_{segmentation_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "patient_id": patient_id,
            "image_file_path": os.path.abspath(image_file_path),
            "segmentation_type": segmentation_type,
            "status": "pending",
            "created_timestamp": datetime.datetime.now().isoformat(),
            "created_by": "patient_database_chat_gui",
            "output_directory": os.path.abspath("images"),
            "expected_output_files": {
                "OAR": f"{patient_id}_oar_mask.nii.gz",
                "BM": f"{patient_id}_bm_mask.nii.gz"
            }.get(segmentation_type, f"{patient_id}_{segmentation_type.lower()}_mask.nii.gz"),
            "description": {
                "OAR": "Organs at Risk segmentation for radiation therapy planning",
                "BM": "Brain Metastases detection and segmentation"
            }.get(segmentation_type, f"{segmentation_type} segmentation task")
        }
        
        # Create task file name
        task_filename = f"{patient_id}.tsk"
        task_file_path = os.path.join(tasks_dir, task_filename)
        
        # Check if task file already exists
        if os.path.exists(task_file_path):
            return {
                "status": "warning",
                "message": f"Task file already exists for patient {patient_id}",
                "task_file": task_file_path,
                "existing_task": True
            }
        
        # Write task file
        with open(task_file_path, 'w') as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "message": f"Segmentation task submitted successfully for patient {patient_id}",
            "task_file": task_file_path,
            "task_id": task_data["task_id"],
            "segmentation_type": segmentation_type,
            "image_file": image_file_path,
            "expected_output": task_data["expected_output_files"],
            "description": task_data["description"]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating segmentation task: {str(e)}"
        }

@tool(show_result=True)
def get_current_datetime() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": now.timestamp()
    }

@tool(show_result=True)
def read_schema() -> str:
    """Read the database schema information from Schema.MD."""
    try:
        with open(SCHEMA_FILE, "r") as f:
            content = f.read()
        return f"Database Schema:\n{content}"
    except Exception as e:
        return f"Error reading schema file: {str(e)}"

@tool(show_result=True)
def query_database(sql_query: str, params: List[Any] = None) -> Dict[str, Any]:
    """
    Execute a SQL query on the patient database.
    
    Args:
        sql_query: The SQL query to execute
        params: Optional list of parameters for the query
        
    Returns:
        Dict containing status and results or error message
    """
    if params is None:
        params = []
    
    try:
        # Check if database exists
        if not os.path.exists(DB_FILE):
            return {
                "status": "error",
                "message": f"Database file '{DB_FILE}' does not exist."
            }
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Return results as dictionaries
        cursor = conn.cursor()
        
        # Execute query
        cursor.execute(sql_query, params)
        
        # Handle different query types
        if sql_query.strip().upper().startswith(("SELECT", "PRAGMA")):
            # For SELECT queries, return the results
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(row))
            
            conn.close()
            return {
                "status": "success",
                "query_type": "select",
                "columns": columns,
                "row_count": len(results),
                "results": results
            }
        else:
            # For other queries (INSERT, UPDATE, DELETE), return affected rows
            affected_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            return {
                "status": "success",
                "query_type": "modify",
                "affected_rows": affected_rows
            }
            
    except sqlite3.Error as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "query": sql_query
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "query": sql_query
        }

@tool(show_result=True)
def insert_patient(
    first_name: str, 
    last_name: str, 
    date_of_birth: str, 
    gender: str,
    address: str = None, 
    phone: str = None, 
    email: str = None, 
    insurance_provider: str = None, 
    insurance_id: str = None
) -> Dict[str, Any]:
    """Insert a new patient record into the database."""
    try:
        # Check if database exists
        if not os.path.exists(DB_FILE):
            return {
                "status": "error",
                "message": f"Database file '{DB_FILE}' does not exist."
            }
        
        # Generate a new patient ID
        patient_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create patient record
        sql = """
        INSERT INTO Patient (
            patient_id, first_name, last_name, date_of_birth, gender,
            address, phone, email, insurance_provider, insurance_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(sql, (
            patient_id, first_name, last_name, date_of_birth, gender,
            address, phone, email, insurance_provider, insurance_id, created_at
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Patient {first_name} {last_name} added successfully",
            "patient_id": patient_id
        }
        
    except sqlite3.Error as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

@tool(show_result=True)
def insert_medical_record(
    patient_id: str,
    primary_diagnosis: str,
    diagnosis_date: str,
    diagnosis_code: str = None,
    cancer_stage: str = None,
    tumor_location: str = None,
    tumor_size: float = None,
    metastasis: str = None,
    allergies: str = None,
    medical_history: str = None,
    medications: str = None,
    previous_treatments: str = None,
    attending_physician: str = None
) -> Dict[str, Any]:
    """Insert a new medical record for a patient."""
    try:
        # Check if database exists
        if not os.path.exists(DB_FILE):
            return {
                "status": "error",
                "message": f"Database file '{DB_FILE}' does not exist."
            }
        
        # Generate a new medical ID
        medical_id = str(uuid.uuid4())
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if patient exists
        cursor.execute("SELECT COUNT(*) FROM Patient WHERE patient_id = ?", (patient_id,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {
                "status": "error",
                "message": f"Patient with ID {patient_id} does not exist."
            }
        
        # Create medical record
        sql = """
        INSERT INTO Medical (
            medical_id, patient_id, primary_diagnosis, diagnosis_date,
            diagnosis_code, cancer_stage, tumor_location, tumor_size,
            metastasis, allergies, medical_history, medications, previous_treatments,
            attending_physician
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(sql, (
            medical_id, patient_id, primary_diagnosis, diagnosis_date,
            diagnosis_code, cancer_stage, tumor_location, tumor_size,
            metastasis, allergies, medical_history, medications, previous_treatments,
            attending_physician
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Medical record added successfully for patient {patient_id}",
            "medical_id": medical_id
        }
        
    except sqlite3.Error as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

@tool(show_result=True)
def insert_treatment_plan(
    patient_id: str,
    medical_id: str,
    treatment_type: str,
    treatment_purpose: str,
    treatment_course: str,
    num_fractions: int,
    dose_per_fraction: float,
    machine_type: str,
    radiation_type: str,
    machine_model: str = None,
    start_date: str = None,
    end_date: str = None,
    oncologist: str = None,
    notes: str = None
) -> Dict[str, Any]:
    """Insert a new treatment plan for a patient."""
    try:
        # Check if database exists
        if not os.path.exists(DB_FILE):
            return {
                "status": "error",
                "message": f"Database file '{DB_FILE}' does not exist."
            }
        
        # Generate a new plan ID
        plan_id = str(uuid.uuid4())
        
        # Calculate total dose
        total_dose = round(num_fractions * dose_per_fraction, 1)
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if patient exists
        cursor.execute("SELECT COUNT(*) FROM Patient WHERE patient_id = ?", (patient_id,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {
                "status": "error",
                "message": f"Patient with ID {patient_id} does not exist."
            }
        
        # Check if medical record exists
        cursor.execute("SELECT COUNT(*) FROM Medical WHERE medical_id = ?", (medical_id,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {
                "status": "error",
                "message": f"Medical record with ID {medical_id} does not exist."
            }
        
        # Create treatment plan
        sql = """
        INSERT INTO Plan (
            plan_id, patient_id, medical_id, treatment_type, treatment_purpose,
            treatment_course, num_fractions, dose_per_fraction, total_dose,
            machine_type, machine_model, radiation_type, start_date, end_date,
            oncologist, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(sql, (
            plan_id, patient_id, medical_id, treatment_type, treatment_purpose,
            treatment_course, num_fractions, dose_per_fraction, total_dose,
            machine_type, machine_model, radiation_type, start_date, end_date,
            oncologist, notes
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Treatment plan added successfully for patient {patient_id}",
            "plan_id": plan_id,
            "total_dose": total_dose
        }
        
    except sqlite3.Error as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

@tool(show_result=True)
def insert_schedule(
    patient_id: str,
    appointment_date: str,
    appointment_time: str,
    appointment_type: str,
    is_treatment: bool,
    plan_id: str = None,
    completed: bool = False,
    duration_minutes: int = 30,
    doctor_name: str = None,
    location: str = None,
    notes: str = None
) -> Dict[str, Any]:
    """Insert a new appointment in the patient's schedule."""
    try:
        # Check if database exists
        if not os.path.exists(DB_FILE):
            return {
                "status": "error",
                "message": f"Database file '{DB_FILE}' does not exist."
            }
        
        # Generate a new schedule ID
        schedule_id = str(uuid.uuid4())
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if patient exists
        cursor.execute("SELECT COUNT(*) FROM Patient WHERE patient_id = ?", (patient_id,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {
                "status": "error",
                "message": f"Patient with ID {patient_id} does not exist."
            }
        
        # Check if plan exists if it's a treatment appointment
        if is_treatment and plan_id:
            cursor.execute("SELECT COUNT(*) FROM Plan WHERE plan_id = ?", (plan_id,))
            if cursor.fetchone()[0] == 0:
                conn.close()
                return {
                    "status": "error",
                    "message": f"Treatment plan with ID {plan_id} does not exist."
                }
        elif is_treatment and not plan_id:
            conn.close()
            return {
                "status": "error",
                "message": "Plan ID is required for treatment appointments."
            }
        
        # Convert booleans to SQLite integers
        is_treatment_int = 1 if is_treatment else 0
        completed_int = 1 if completed else 0
        
        # Create schedule record
        sql = """
        INSERT INTO Schedule (
            schedule_id, patient_id, plan_id, appointment_date, appointment_time,
            appointment_type, is_treatment, completed, duration_minutes,
            doctor_name, location, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(sql, (
            schedule_id, patient_id, plan_id, appointment_date, appointment_time,
            appointment_type, is_treatment_int, completed_int, duration_minutes,
            doctor_name, location, notes, created_at
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Appointment scheduled successfully for {appointment_date} at {appointment_time}",
            "schedule_id": schedule_id
        }
        
    except sqlite3.Error as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

@tool(show_result=True)
def update_patient(
    patient_id: str,
    first_name: str = None,
    last_name: str = None,
    date_of_birth: str = None,
    gender: str = None,
    address: str = None,
    phone: str = None,
    email: str = None,
    insurance_provider: str = None,
    insurance_id: str = None
) -> Dict[str, Any]:
    """Update an existing patient record."""
    try:
        # Check if database exists
        if not os.path.exists(DB_FILE):
            return {
                "status": "error",
                "message": f"Database file '{DB_FILE}' does not exist."
            }
        
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if patient exists
        cursor.execute("SELECT COUNT(*) FROM Patient WHERE patient_id = ?", (patient_id,))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return {
                "status": "error",
                "message": f"Patient with ID {patient_id} does not exist."
            }
        
        # Build update query dynamically based on provided fields
        update_fields = []
        params = []
        
        field_mappings = {
            "first_name": first_name,
            "last_name": last_name,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "address": address,
            "phone": phone,
            "email": email,
            "insurance_provider": insurance_provider,
            "insurance_id": insurance_id
        }
        
        for field, value in field_mappings.items():
            if value is not None:
                update_fields.append(f"{field} = ?")
                params.append(value)
        
        if not update_fields:
            conn.close()
            return {
                "status": "error",
                "message": "No fields provided for update."
            }
        
        # Add patient_id to params
        params.append(patient_id)
        
        # Create and execute update query
        sql = f"UPDATE Patient SET {', '.join(update_fields)} WHERE patient_id = ?"
        cursor.execute(sql, params)
        
        if cursor.rowcount == 0:
            conn.close()
            return {
                "status": "warning",
                "message": "No changes made to the record."
            }
        
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": f"Patient record updated successfully",
            "updated_fields": list(field for field, value in field_mappings.items() if value is not None)
        }
        
    except sqlite3.Error as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}"
        }

# NIfTI Image Viewer Widget
class NiftiViewer(QWidget):
    """Widget for displaying NIfTI medical images with slice navigation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image_data = None
        self.current_file_path = None
        self.current_mask_data = None
        self.current_mask_path = None
        self.current_slice = 0
        self.max_slices = 0
        self.show_overlay = True
        self.overlay_alpha = 0.5
        
        # Auto-play functionality
        self.auto_play_timer = QTimer()
        self.auto_play_timer.timeout.connect(self.auto_play_next_slice)
        self.auto_play_speed = 200  # milliseconds between slices
        self.is_auto_playing = False
        self.manual_navigation_time = 0  # Track manual navigation to avoid conflicts
        
        # 3D/2D view state
        self.is_3d_view = False
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the NIfTI viewer UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        self.title_label = QLabel("Medical Image Viewer")
        title_font = QApplication.font()
        title_font.setPointSize(14)
        title_font.setWeight(QFont.Weight.Bold)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background-color: #2d2d2d;
                border-radius: 8px;
                padding: 8px;
                border: 1px solid #3d3d3d;
            }
        """)
        
        # Create stacked widget to hold 2D and 3D views
        self.view_stack = QStackedWidget()
        
        # 2D view - Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                color: #888888;
            }
        """)
        
        # 3D view - Use simplified software-based viewer to avoid OpenGL conflicts
        # Create a simple volume projection viewer instead of full OpenGL
        self.volume_viewer_3d = Simple3DViewer()
        self.volume_viewer_3d.setMinimumSize(400, 400)
        
        # Add both views to the stacked widget
        self.view_stack.addWidget(self.image_label)  # Index 0: 2D view
        self.view_stack.addWidget(self.volume_viewer_3d)  # Index 1: 3D view
        self.view_stack.setCurrentIndex(0)  # Start with 2D view
        self.image_label.setText("No image loaded")
        
        # Slice navigation controls
        controls_layout = QHBoxLayout()
        
        # Slice info
        self.slice_info_label = QLabel("Slice: 0 / 0")
        self.slice_info_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        
        # Slice slider
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3d3d3d;
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
        """)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        
        # Navigation buttons
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        
        button_style = """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
        """
        self.prev_button.setStyleSheet(button_style)
        self.next_button.setStyleSheet(button_style)
        
        self.prev_button.clicked.connect(self.previous_slice)
        self.next_button.clicked.connect(self.next_slice)
        
        # Image info display
        self.info_label = QLabel("No image information available")
        self.info_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background-color: #2d2d2d;
                border-radius: 6px;
                padding: 8px;
                border: 1px solid #3d3d3d;
            }
        """)
        self.info_label.setWordWrap(True)
        
        # Overlay controls - smaller button
        self.overlay_checkbox = QPushButton("🎯")
        self.overlay_checkbox.setCheckable(True)
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.setMaximumWidth(40)
        self.overlay_checkbox.setToolTip("Toggle segmentation overlay")
        overlay_button_style = """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:checked {
                background-color: #28a745;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
        """
        self.overlay_checkbox.setStyleSheet(overlay_button_style)
        
        # Connect the button - use toggled signal for checkable buttons
        self.overlay_checkbox.toggled.connect(self.on_overlay_toggled)
        
        # Auto-play button
        self.auto_play_button = QPushButton("▶")
        self.auto_play_button.setCheckable(True)
        self.auto_play_button.setMaximumWidth(40)
        self.auto_play_button.setToolTip("Auto-play through slices")
        self.auto_play_button.setStyleSheet(overlay_button_style)  # Use same style as overlay button
        self.auto_play_button.toggled.connect(self.toggle_auto_play)
        
        # 3D/2D view toggle button
        self.view_toggle_button = QPushButton("🧊")
        self.view_toggle_button.setCheckable(True)
        self.view_toggle_button.setMaximumWidth(40)
        self.view_toggle_button.setToolTip("Switch to 3D view")
        self.view_toggle_button.setStyleSheet(overlay_button_style)
        self.view_toggle_button.toggled.connect(self.toggle_view_mode)
        self.view_toggle_button.setEnabled(True)  # Always enabled with software 3D viewer
        
        # Assemble controls layout
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.slice_info_label)
        controls_layout.addWidget(self.slice_slider, 1)
        controls_layout.addWidget(self.auto_play_button)
        controls_layout.addWidget(self.view_toggle_button)
        controls_layout.addWidget(self.overlay_checkbox)
        controls_layout.addWidget(self.next_button)
        
        # Assemble main layout
        layout.addWidget(self.title_label)
        layout.addWidget(self.view_stack, 1)  # Use stacked widget instead of image_label
        layout.addLayout(controls_layout)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
        
        # Initially disable controls
        self.update_controls_state()
        
    def load_image(self, file_path: str):
        """Load a NIfTI image file."""
        try:
            import nibabel as nib
            import numpy as np
            from PIL import Image
            import io
            
            # Load the NIfTI image
            img = nib.load(file_path)
            self.current_image_data = img.get_fdata()
            self.current_file_path = file_path
            
            # Handle different image dimensions
            if len(self.current_image_data.shape) >= 3:
                self.max_slices = self.current_image_data.shape[2]
                self.current_slice = self.max_slices // 2  # Start from middle slice
            else:
                self.max_slices = 1
                self.current_slice = 0
            
            # Update UI
            self.slice_slider.setMaximum(self.max_slices - 1)
            self.slice_slider.setValue(self.current_slice)
            
            # Update info display
            shape_str = " × ".join(map(str, self.current_image_data.shape))
            self.info_label.setText(f"File: {os.path.basename(file_path)}\nShape: {shape_str}\nData type: {self.current_image_data.dtype}")
            
            # Display the current slice
            self.display_current_slice()
            self.update_controls_state()
            
            # Load volume into 3D viewer
            if hasattr(self, 'volume_viewer_3d'):
                self.volume_viewer_3d.load_volume(self.current_image_data)
            
            # Show the viewer now that an image is loaded
            self.setVisible(True)
            
            return True
            
        except Exception as e:
            self.image_label.setText(f"Error loading image:\n{str(e)}")
            self.info_label.setText("Failed to load image")
            return False
    
    def display_current_slice(self):
        """Display the current slice of the loaded image with optional overlay."""
        if self.current_image_data is None:
            return
            
        try:
            import numpy as np
            from PIL import Image
            import io
            
            # Get current slice data
            if len(self.current_image_data.shape) >= 3:
                slice_data = self.current_image_data[:, :, self.current_slice]
            else:
                slice_data = self.current_image_data
            
            # Normalize the data to 0-255 range with better contrast
            slice_data = slice_data.astype(np.float64)
            
            # Use percentile-based normalization for better contrast
            p2, p98 = np.percentile(slice_data, (2, 98))
            slice_data = np.clip(slice_data, p2, p98)
            
            slice_min, slice_max = np.min(slice_data), np.max(slice_data)
            if slice_max > slice_min:
                slice_data = (slice_data - slice_min) / (slice_max - slice_min) * 255
            else:
                slice_data = np.ones_like(slice_data) * 128  # Gray if no variation
            
            slice_data = slice_data.astype(np.uint8)
            
            # Create base image
            if self.current_mask_data is not None and self.show_overlay:
                # Create RGB image for overlay
                rgb_slice = np.stack([slice_data, slice_data, slice_data], axis=-1)
                
                # Get mask slice
                if len(self.current_mask_data.shape) >= 3:
                    mask_slice = self.current_mask_data[:, :, self.current_slice]
                else:
                    mask_slice = self.current_mask_data
                
                # Apply colormap to mask
                rgb_slice = self._apply_mask_overlay(rgb_slice, mask_slice)
                pil_image = Image.fromarray(rgb_slice, mode='RGB')
            else:
                # Grayscale image without overlay
                pil_image = Image.fromarray(slice_data, mode='L')
            
            # Rotate 90 degrees clockwise
            pil_image = pil_image.rotate(-90, expand=True)
            
            # Resize to fit display (maintain aspect ratio)
            display_size = (400, 400)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to QPixmap
            byte_array = io.BytesIO()
            pil_image.save(byte_array, format='PNG')
            byte_array.seek(0)
            
            pixmap = QPixmap()
            success = pixmap.loadFromData(byte_array.getvalue())
            
            # Display in label
            self.image_label.setPixmap(pixmap)
            
            # Update slice info
            mask_exists = self.current_mask_data is not None
            has_overlay = mask_exists and self.show_overlay
            overlay_info = f" (Overlay: {'ON' if has_overlay else 'OFF'})"
            self.slice_info_label.setText(f"Slice: {self.current_slice + 1} / {self.max_slices}{overlay_info}")
            
            # Debug info (commented out for clean operation)
            # print(f"DEBUG: Display update - mask exists: {mask_exists}, show_overlay: {self.show_overlay}, has_overlay: {has_overlay}")
            # print(f"DEBUG: Button checked state: {self.overlay_checkbox.isChecked()}")
            
        except Exception as e:
            self.image_label.setText(f"Error displaying slice:\n{str(e)}")
    
    def on_slice_changed(self, value):
        """Handle slice slider value change."""
        self.current_slice = value
        self.display_current_slice()
    
    def previous_slice(self):
        """Navigate to previous slice."""
        self.manual_navigation_time = time.time()
        if self.current_slice > 0:
            self.current_slice -= 1
            self.slice_slider.setValue(self.current_slice)
    
    def next_slice(self):
        """Navigate to next slice."""
        self.manual_navigation_time = time.time()
        if self.current_slice < self.max_slices - 1:
            self.current_slice += 1
            self.slice_slider.setValue(self.current_slice)
    
    def update_controls_state(self):
        """Update the enabled/disabled state of controls."""
        has_image = self.current_image_data is not None
        has_multiple_slices = self.max_slices > 1
        
        self.slice_slider.setEnabled(has_image and has_multiple_slices)
        self.prev_button.setEnabled(has_image and has_multiple_slices and self.current_slice > 0)
        self.next_button.setEnabled(has_image and has_multiple_slices and self.current_slice < self.max_slices - 1)
        self.auto_play_button.setEnabled(has_image and has_multiple_slices)
    
    def clear_image(self):
        """Clear the current image display."""
        # Stop auto-play when clearing image
        if self.is_auto_playing:
            self.auto_play_button.setChecked(False)
            self.toggle_auto_play(False)
            
        self.current_image_data = None
        self.current_file_path = None
        self.current_slice = 0
        self.max_slices = 0
        
        self.image_label.clear()
        self.image_label.setText("No image loaded")
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_info_label.setText("Slice: 0 / 0")
        self.info_label.setText("No image information available")
        
        self.update_controls_state()
        
        # Hide the viewer when no image is loaded
        self.setVisible(False)
    
    def load_mask(self, file_path: str):
        """Load a segmentation mask for overlay."""
        try:
            import nibabel as nib
            import numpy as np
            
            # Load the mask
            mask_img = nib.load(file_path)
            mask_data_raw = mask_img.get_fdata()
            self.current_mask_data = mask_data_raw.astype(np.uint8)
            self.current_mask_path = file_path
            
            # Verify mask dimensions match current image
            if self.current_image_data is not None:
                if self.current_mask_data.shape != self.current_image_data.shape:
                    self.current_mask_data = None
                    self.current_mask_path = None
                    return False, f"Mask dimensions {self.current_mask_data.shape} don't match image dimensions {self.current_image_data.shape}"
            
            # Ensure overlay is enabled when mask is loaded
            self.show_overlay = True
            # Block signals to prevent triggering the toggled slot when setting state programmatically
            self.overlay_checkbox.blockSignals(True)
            self.overlay_checkbox.setChecked(True)
            self.overlay_checkbox.blockSignals(False)
            
            # Update tooltip
            self.overlay_checkbox.setToolTip("Hide segmentation overlay")
            
            # Debug info (commented out for clean operation)
            # print(f"DEBUG: Mask loaded, overlay enabled: {self.show_overlay}")
            # print(f"DEBUG: Button checked after mask load: {self.overlay_checkbox.isChecked()}")
            
            # Refresh display
            self.display_current_slice()
            self.update_controls_state()
            
            # Load mask into 3D viewer
            if hasattr(self, 'volume_viewer_3d'):
                self.volume_viewer_3d.load_mask(self.current_mask_data)
                if self.is_3d_view:
                    self.volume_viewer_3d.toggle_mask_overlay(self.show_overlay)
            
            return True, f"Mask loaded successfully from {os.path.basename(file_path)}"
            
        except Exception as e:
            self.current_mask_data = None
            self.current_mask_path = None
            return False, f"Error loading mask: {str(e)}"
    
    def _apply_mask_overlay(self, rgb_image, mask_slice):
        """Apply colorized mask overlay to RGB image."""
        import numpy as np
        
        # Define colors for different mask labels (RGB format) - supports up to 30 labels
        colors = {
            0: [0, 0, 0],        # Background (transparent)
            1: [255, 0, 0],      # Red
            2: [0, 255, 0],      # Green  
            3: [0, 0, 255],      # Blue
            4: [255, 255, 0],    # Yellow
            5: [255, 0, 255],    # Magenta
            6: [0, 255, 255],    # Cyan
            7: [128, 0, 0],      # Dark red
            8: [0, 128, 0],      # Dark green
            9: [0, 0, 128],      # Dark blue
            10: [255, 128, 0],   # Orange
            11: [255, 0, 128],   # Pink
            12: [128, 255, 0],   # Lime
            13: [0, 128, 255],   # Light blue
            14: [128, 0, 255],   # Purple
            15: [255, 128, 128], # Light red
            16: [128, 255, 128], # Light green
            17: [128, 128, 255], # Light blue
            18: [192, 192, 0],   # Olive
            19: [192, 0, 192],   # Purple
            20: [0, 192, 192],   # Teal
            21: [255, 192, 0],   # Gold
            22: [192, 255, 0],   # Yellow-green
            23: [0, 255, 192],   # Spring green
            24: [192, 0, 255],   # Violet
            25: [255, 0, 192],   # Rose
            26: [64, 64, 64],    # Dark gray
            27: [192, 192, 192], # Light gray
            28: [128, 64, 0],    # Brown
            29: [0, 128, 64],    # Forest green
            30: [64, 0, 128],    # Indigo
        }
        
        result = rgb_image.copy().astype(np.float32)
        
        # Apply each mask label with its color
        unique_labels = np.unique(mask_slice)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            mask_region = mask_slice == label
            num_voxels = np.sum(mask_region)
            if num_voxels > 0:
                color = colors.get(int(label), [255, 255, 255])  # Default to white
                
                # Blend the color with alpha transparency
                alpha = self.overlay_alpha
                for channel in range(3):
                    result[mask_region, channel] = (
                        alpha * color[channel] + 
                        (1 - alpha) * result[mask_region, channel]
                    )
        
        return result.astype(np.uint8)
    
    def on_overlay_toggled(self, checked):
        """Handle overlay button toggle signal."""
        self.show_overlay = checked
        
        # Update both 2D and 3D views
        if self.is_3d_view:
            # Update 3D view
            if hasattr(self, 'volume_viewer_3d'):
                self.volume_viewer_3d.toggle_mask_overlay(self.show_overlay)
        else:
            # Update 2D view
            self.display_current_slice()
        
        # Update overlay controls
        self.update_overlay_controls()
    
    def toggle_auto_play(self, checked):
        """Toggle auto-play animation on/off."""
        self.is_auto_playing = checked
        
        if self.is_auto_playing:
            # Start auto-play
            self.auto_play_button.setText("⏸")
            self.auto_play_button.setToolTip("Pause auto-play")
            self.auto_play_timer.start(self.auto_play_speed)
        else:
            # Stop auto-play
            self.auto_play_button.setText("▶")
            self.auto_play_button.setToolTip("Auto-play through slices")
            self.auto_play_timer.stop()
    
    def auto_play_next_slice(self):
        """Move to next slice during auto-play animation."""
        if self.current_image_data is not None and self.max_slices > 1:
            # Skip auto-advance if user recently navigated manually (within 500ms)
            current_time = time.time()
            if current_time - self.manual_navigation_time < 0.5:
                return
                
            # Move to next slice, loop back to beginning when reaching the end
            self.current_slice = (self.current_slice + 1) % self.max_slices
            self.slice_slider.setValue(self.current_slice)
            self.display_current_slice()
    
    def set_auto_play_speed(self, speed_ms):
        """Set the auto-play speed in milliseconds between slices."""
        self.auto_play_speed = speed_ms
        if self.is_auto_playing:
            # Restart timer with new interval
            self.auto_play_timer.start(self.auto_play_speed)
    
    def toggle_view_mode(self, checked):
        """Toggle between 2D and 3D view modes."""
        self.is_3d_view = checked
        
        if self.is_3d_view:
            # Switch to 3D view
            self.view_stack.setCurrentIndex(1)
            self.view_toggle_button.setText("📋")
            self.view_toggle_button.setToolTip("Switch to 2D view")
            
            # Load current data into 3D viewer
            if self.current_image_data is not None:
                self.volume_viewer_3d.load_volume(self.current_image_data)
                if self.current_mask_data is not None:
                    self.volume_viewer_3d.load_mask(self.current_mask_data)
                    self.volume_viewer_3d.toggle_mask_overlay(self.show_overlay)
            
            # Disable slice controls for 3D view
            self.slice_slider.setEnabled(False)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.auto_play_button.setEnabled(False)
            
        else:
            # Switch to 2D view
            self.view_stack.setCurrentIndex(0)
            self.view_toggle_button.setText("🧊")
            self.view_toggle_button.setToolTip("Switch to 3D view")
            
            # Re-enable slice controls for 2D view
            self.update_controls_state()
            
        # Update view-specific overlay controls
        self.update_overlay_controls()
    
    def update_overlay_controls(self):
        """Update overlay controls based on current view mode."""
        if self.is_3d_view:
            # In 3D view, overlay control affects 3D rendering
            self.overlay_checkbox.setToolTip("Toggle 3D mask overlay")
            if hasattr(self, 'volume_viewer_3d'):
                self.volume_viewer_3d.toggle_mask_overlay(self.show_overlay)
        else:
            # In 2D view, overlay control affects 2D slice display
            tooltip = "Hide segmentation overlay" if self.show_overlay else "Show segmentation overlay"
            self.overlay_checkbox.setToolTip(tooltip)
    
    def clear_mask(self):
        """Clear the current segmentation mask."""
        self.current_mask_data = None
        self.current_mask_path = None
        self.display_current_slice()
        self.update_controls_state()

# Simple Software-based 3D Viewer (No OpenGL)
class Simple3DViewer(QLabel):
    """Software-based 3D volume viewer using maximum intensity projection."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume_data = None
        self.mask_data = None
        self.rotation_angle = 0
        self.show_mask_overlay = False
        
        # Set initial appearance
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                border: 2px solid #3d3d3d;
                border-radius: 8px;
                color: #888888;
            }
        """)
        self.setText("3D Volume Viewer\n(Software Rendering)")
        
        # Enable mouse tracking for rotation
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        
    def load_volume(self, volume_data):
        """Load volume data and create 3D projection."""
        if volume_data is None:
            return
            
        self.volume_data = volume_data
        self.render_volume()
        
    def load_mask(self, mask_data):
        """Load mask data for overlay."""
        if mask_data is None:
            return
            
        self.mask_data = mask_data
        if self.show_mask_overlay:
            self.render_volume()
        
    def toggle_mask_overlay(self, show):
        """Toggle mask overlay display."""
        self.show_mask_overlay = show
        if self.volume_data is not None:
            self.render_volume()
            
    def render_volume(self):
        """Create a 3D-looking projection of the volume data."""
        if self.volume_data is None:
            return
            
        try:
            import numpy as np
            from PIL import Image
            
            # Simple discrete rotation (cleaner, no blending artifacts)
            # Normalize rotation angle to select view
            angle_step = (self.rotation_angle % 360) // 45  # 8 discrete steps (every 45°)
            
            # Select projection based on rotation angle
            if angle_step in [0, 1]:  # 0° to 90° - front view
                projection = np.max(self.volume_data, axis=2)  # Z-axis projection
            elif angle_step in [2, 3]:  # 90° to 180° - side view
                projection = np.max(self.volume_data, axis=0)  # X-axis projection
            elif angle_step in [4, 5]:  # 180° to 270° - back view (flipped front)
                projection = np.fliplr(np.max(self.volume_data, axis=2))  # Flipped Z-axis
            else:  # angle_step in [6, 7] - 270° to 360° - other side view
                projection = np.fliplr(np.max(self.volume_data, axis=0))  # Flipped X-axis
            
            # Normalize to 0-255
            projection = (projection - np.min(projection)) / (np.max(projection) - np.min(projection)) * 255
            projection = projection.astype(np.uint8)
            
            # Add mask overlay if enabled
            if self.show_mask_overlay and self.mask_data is not None:
                # Create mask projection using same angle logic as volume
                if angle_step in [0, 1]:  # 0° to 90° - front view
                    mask_proj = np.max(self.mask_data, axis=2)  # Z-axis projection
                elif angle_step in [2, 3]:  # 90° to 180° - side view
                    mask_proj = np.max(self.mask_data, axis=0)  # X-axis projection
                elif angle_step in [4, 5]:  # 180° to 270° - back view (flipped front)
                    mask_proj = np.fliplr(np.max(self.mask_data, axis=2))  # Flipped Z-axis
                else:  # angle_step in [6, 7] - 270° to 360° - other side view
                    mask_proj = np.fliplr(np.max(self.mask_data, axis=0))  # Flipped X-axis
                
                # Convert to RGB and overlay mask in red
                rgb_projection = np.stack([projection, projection, projection], axis=-1)
                mask_overlay = mask_proj > 0
                rgb_projection[mask_overlay, 0] = np.minimum(255, rgb_projection[mask_overlay, 0] + 100)  # Add red
                
                # Convert to PIL image
                pil_image = Image.fromarray(rgb_projection, mode='RGB')
            else:
                # Grayscale projection
                pil_image = Image.fromarray(projection, mode='L')
            
            # Rotate 90 degrees counterclockwise to match 2D viewer orientation
            pil_image = pil_image.rotate(90, expand=True)
            
            # Resize to fit display
            display_size = (380, 380)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to QPixmap and display
            import io
            byte_array = io.BytesIO()
            pil_image.save(byte_array, format='PNG')
            byte_array.seek(0)
            
            pixmap = QPixmap()
            pixmap.loadFromData(byte_array.getvalue())
            self.setPixmap(pixmap)
            
        except Exception as e:
            self.setText(f"3D Rendering Error:\n{str(e)}")
    
    def mousePressEvent(self, event):
        """Handle mouse press for rotation."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.position()
            # Change cursor to indicate dragging
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
    def mouseMoveEvent(self, event):
        """Handle mouse movement for smooth horizontal rotation."""
        if self.last_mouse_pos is not None:
            dx = event.position().x() - self.last_mouse_pos.x()
            
            # Update rotation angle with improved sensitivity
            rotation_sensitivity = 1.0  # Degrees per pixel
            self.rotation_angle += dx * rotation_sensitivity
            
            # Keep angle in 0-360 range
            self.rotation_angle = self.rotation_angle % 360
            
            self.last_mouse_pos = event.position()
            
            # Re-render with new angle for smooth rotation
            if self.volume_data is not None:
                self.render_volume()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None
            # Reset cursor
            self.setCursor(Qt.CursorShape.OpenHandCursor)
    
    def enterEvent(self, event):
        """Handle mouse enter - show hand cursor."""
        if self.volume_data is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave - reset cursor."""
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)

# 3D OpenGL Volume Viewer (Disabled - using Simple3DViewer instead)
# VolumeViewer3D class has been disabled to avoid OpenGL import conflicts
# We're using Simple3DViewer instead which provides software-based 3D rendering

# Collapsible Widget for Tool Calls
class CollapsibleWidget(QWidget):
    """A collapsible widget that can show/hide its content."""
    
    def __init__(self, title: str, content: str, parent=None):
        super().__init__(parent)
        self.content_visible = False
        self.init_ui(title, content)
        
    def init_ui(self, title: str, content: str):
        """Initialize the collapsible widget UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Toggle button
        self.toggle_button = QPushButton(f"> {title}")
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px 12px;
                background-color: #333333;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 6px;
                font-weight: 500;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #444444;
                color: #ffffff;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_content)
        
        # Content area
        self.content_widget = QTextBrowser()
        self.content_widget.setHtml(content)
        self.content_widget.setMaximumHeight(200)
        self.content_widget.setStyleSheet("""
            QTextBrowser {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 10px;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;\n            font-size: 13px;
                font-size: 11px;
            }
        """)
        self.content_widget.setVisible(False)
        
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_widget)
        self.setLayout(layout)
    
    def toggle_content(self):
        """Toggle the visibility of the content."""
        self.content_visible = not self.content_visible
        self.content_widget.setVisible(self.content_visible)
        
        # Update button text
        button_text = self.toggle_button.text()
        if self.content_visible:
            self.toggle_button.setText(button_text.replace(">", "v"))
        else:
            self.toggle_button.setText(button_text.replace("v", ">"))


# Removed MessageBlock class - now using HTML-based approach


# Removed ToolBlock class - now using HTML-based approach


# Agent Worker Thread
class AgentWorker(QObject):
    """Worker class for running agent queries in a separate thread."""
    
    response_ready = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    thinking_update = pyqtSignal(str)
    tool_call_started = pyqtSignal(str, str)  # tool_name, tool_args
    tool_call_finished = pyqtSignal(str, str)  # tool_name, tool_result
    partial_response = pyqtSignal(str)  # For streaming output
    stream_started = pyqtSignal()  # When streaming begins
    stream_finished = pyqtSignal()  # When streaming completes
    
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        
    def process_query(self, query: str):
        """Process a user query with the agent."""
        try:
            # Simple query classification for better user feedback
            if any(x in query.lower() for x in ["how many", "count", "number of"]):
                self.thinking_update.emit("I'll need to run a COUNT query on the database.")
            elif any(x in query.lower() for x in ["show", "list", "display", "get"]):
                self.thinking_update.emit("I'll need to retrieve records from the database.")
            elif any(x in query.lower() for x in ["add", "create", "insert", "new"]):
                self.thinking_update.emit("I'll need to create a new record in the database.")
            elif any(x in query.lower() for x in ["update", "change", "modify", "edit"]):
                self.thinking_update.emit("I'll need to update an existing record in the database.")
            
            # Get response from agent with streaming
            self.stream_started.emit()
            response = self.agent.run(query, stream=True)
            
            # Handle streaming response
            is_streaming = hasattr(response, '__iter__') and not isinstance(response, str)
            if is_streaming:
                # This is a streaming response
                accumulated_text = ""
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        accumulated_text += chunk.content
                        self._detect_and_emit_tool_calls(chunk.content)
                        self.partial_response.emit(accumulated_text)
                    elif isinstance(chunk, str):
                        accumulated_text += chunk
                        self._detect_and_emit_tool_calls(chunk)
                        self.partial_response.emit(accumulated_text)
                
                self.stream_finished.emit()
                response_text = accumulated_text
                # For streaming responses, don't emit response_ready as streaming handles the display
                return
            else:
                # Non-streaming response fallback
                if hasattr(response, 'content'):
                    response_text = response.content
                elif isinstance(response, str):
                    response_text = response
                else:
                    response_text = str(response)
                self.stream_finished.emit()
            
            # Extract tool calls and results from messages if available
            if hasattr(response, 'messages') and response.messages:
                self._process_tool_calls(response.messages)
            
            # Extract the response content
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Clean up response to remove duplicate tool result JSON
            original_text = response_text
            response_text = self._clean_response_text(response_text)
            
            # Debug: Print if significant cleaning occurred
            if len(original_text) > len(response_text) + 50:  # If we removed substantial content
                print(f"[DEBUG] Cleaned response - removed {len(original_text) - len(response_text)} characters")
                print(f"[DEBUG] Original: {original_text[:100]}...")
                print(f"[DEBUG] Cleaned: {response_text[:100]}...")
                
            self.response_ready.emit(response_text)
            
        except Exception as e:
            self.error_occurred.emit(f"Error processing query: {str(e)}")
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean up response text to remove duplicate tool result JSON."""
        import re
        import json
        
        # More aggressive JSON removal patterns
        # Remove complete JSON objects that contain tool result patterns
        json_patterns = [
            # Complete JSON objects with status field
            r'\{[^{}]*[\'"]status[\'"]:\s*[\'"][^\'"}]*[\'"][^{}]*\}',
            # JSON objects with row_count field  
            r'\{[^{}]*[\'"]row_count[\'"]:\s*\d+[^{}]*\}',
            # JSON objects with query_type field
            r'\{[^{}]*[\'"]query_type[\'"]:\s*[\'"][^\'"}]*[\'"][^{}]*\}',
            # JSON objects with results array
            r'\{[^{}]*[\'"]results[\'"]:\s*\[[^\]]*\][^{}]*\}',
            # JSON objects with columns field
            r'\{[^{}]*[\'"]columns[\'"]:\s*\[[^\]]*\][^{}]*\}',
        ]
        
        # Apply aggressive JSON removal
        for pattern in json_patterns:
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove any remaining JSON-like structures at the beginning
        response_text = re.sub(r'^\s*\{[^}]*\}\s*', '', response_text, flags=re.MULTILINE)
        
        # Split into lines and process each one
        lines = response_text.split('\n')
        cleaned_lines = []
        skip_next_lines = False
        
        for i, line in enumerate(lines):
            original_line = line
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are clearly JSON objects
            if line.startswith('{') and '}' in line:
                try:
                    # Try to parse as JSON to confirm it's a JSON object
                    if line.endswith('}'):
                        json_obj = json.loads(line)
                        # If it parses and has tool result fields, skip it
                        if isinstance(json_obj, dict) and any(key in json_obj for key in 
                            ['status', 'row_count', 'results', 'query_type', 'columns']):
                            continue
                    else:
                        # Might be the start of a multi-line JSON, skip until we find the end
                        if any(key in line for key in ['status', 'row_count', 'query_type', 'results', 'columns']):
                            skip_next_lines = True
                            continue
                except json.JSONDecodeError:
                    pass
            
            # Skip lines if we're in the middle of a multi-line JSON
            if skip_next_lines:
                if '}' in line:
                    skip_next_lines = False
                continue
            
            # Remove any remaining JSON fragments from within the line
            line = re.sub(r'\{[^}]*[\'"](?:status|row_count|query_type|results|columns)[\'"][^}]*\}', '', line)
            
            # Skip lines that still contain JSON-like patterns
            if re.search(r'\{.*[\'"](?:status|row_count|query_type|results)[\'"].*\}', line):
                continue
            
            # Only add non-empty, clean lines
            line = line.strip()
            if line and not line.startswith('{'):
                cleaned_lines.append(line)
        
        # Join back and clean up whitespace
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove any leftover JSON fragments at the start of the text
        cleaned_text = re.sub(r'^\s*\{[^}]*\}\s*', '', cleaned_text)
        
        # Clean up multiple blank lines
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _detect_and_emit_tool_calls(self, text_chunk: str):
        """Detect tool calls in streaming text and emit signals."""
        try:
            import re
            import json
            
            # Track already processed tool results to avoid duplicates
            if not hasattr(self, '_processed_tool_results'):
                self._processed_tool_results = set()
            
            # Look for JSON objects that look like tool results (handle both single and double quotes)
            json_patterns = [
                r'\{[^}]*[\'"]status[\'"][^}]*[\'"]success[\'"][^}]*\}',
                r'\{[^}]*[\'"]query_type[\'"][^}]*\}',
                r'\{[^}]*[\'"]results[\'"][^}]*\}',
            ]
            
            matches = []
            for pattern in json_patterns:
                matches.extend(re.findall(pattern, text_chunk))
            
            for match in matches:
                # Skip if we've already processed this exact tool result
                if match in self._processed_tool_results:
                    continue
                    
                self._processed_tool_results.add(match)
                
                try:
                    # Try to parse as JSON - handle both single and double quotes
                    normalized_json = match.replace("'", '"')
                    # Also handle Python-style boolean values
                    normalized_json = normalized_json.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                    
                    result_obj = json.loads(normalized_json)
                    
                    # Check if this looks like a database tool result
                    if 'query_type' in result_obj:
                        tool_name = 'query_database'
                        sql_hint = result_obj.get('query_type', 'unknown')
                        columns = result_obj.get('columns', [])
                        row_count = result_obj.get('row_count', 0)
                        
                        tool_args = f"sql_query='SELECT {', '.join(columns) if columns else '*'} FROM table' (returned {row_count} rows)"
                        tool_result = match
                        
                        # Emit both started and finished for this tool
                        self.tool_call_started.emit(tool_name, tool_args)
                        self.tool_call_finished.emit(tool_name, tool_result)
                    
                    elif 'status' in result_obj and result_obj.get('status') == 'success':
                        # Generic successful tool result
                        tool_name = 'database_operation'
                        tool_args = 'Database query executed'
                        tool_result = match
                        
                        self.tool_call_started.emit(tool_name, tool_args)
                        self.tool_call_finished.emit(tool_name, tool_result)
                        
                except (json.JSONDecodeError, ValueError) as e:
                    # If JSON parsing fails, still try to detect tool-like patterns
                    if 'status' in match and 'success' in match:
                        tool_name = 'database_query'
                        tool_args = 'SQL query execution'
                        tool_result = match
                        
                        self.tool_call_started.emit(tool_name, tool_args)
                        self.tool_call_finished.emit(tool_name, tool_result)
                    continue
                    
        except Exception as e:
            # Silently handle errors to avoid breaking streaming
            pass
    
    def _process_tool_calls(self, messages):
        """Extract and emit tool calls from agent messages."""
        try:
            # Look for tool-related content in messages
            for i, message in enumerate(messages):
                # Check for tool calls (various possible formats)
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call, 'function'):
                            tool_name = tool_call.function.name
                            tool_args = str(tool_call.function.arguments)
                            self.tool_call_started.emit(tool_name, tool_args)
                
                # Check for tool role messages
                if hasattr(message, 'role'):
                    if message.role == 'tool' and hasattr(message, 'content'):
                        tool_name = getattr(message, 'name', getattr(message, 'tool_name', 'unknown_tool'))
                        tool_result = str(message.content)
                        self.tool_call_finished.emit(tool_name, tool_result)
                    
                    # Also check assistant messages that might contain tool information
                    elif message.role == 'assistant' and hasattr(message, 'content') and message.content:
                        content = str(message.content)
                        # Simple heuristic to detect tool-like content
                        if 'Tool Calls' in content or 'query_database' in content or 'read_schema' in content:
                            # This is a fallback for when tool info is embedded in content
                            lines = content.split('\n')
                            for line in lines:
                                if 'query_database(' in line or 'read_schema(' in line:
                                    # Extract tool name and basic args
                                    if '(' in line and ')' in line:
                                        tool_part = line.split('(')[0].strip('• ').strip()
                                        args_part = line[line.find('(')+1:line.rfind(')')]
                                        self.tool_call_started.emit(tool_part, args_part)
                        
        except Exception as e:
            # Silently handle tool parsing errors to avoid breaking the main flow
            pass

class ChatWidget(QWidget):
    """Custom chat widget with message display and input."""
    
    def __init__(self):
        super().__init__()
        self.markdown_renderer = MarkdownRenderer()
        self.last_tool_call = None
        self.tool_details = {}  # Store detailed tool information
        self.init_ui()
        
    def init_ui(self):
        """Initialize the chat UI components."""
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Add toolbar for global controls
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)
        
        # Collapse All button
        self.collapse_all_btn = QPushButton("Collapse All")
        self.collapse_all_btn.setFixedSize(100, 30)
        self.collapse_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #666666;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #555555;
                border-color: #777777;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        self.collapse_all_btn.clicked.connect(self.collapse_all_blocks)
        
        # Expand All button
        self.expand_all_btn = QPushButton("Expand All")
        self.expand_all_btn.setFixedSize(100, 30)
        self.expand_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: 1px solid #0078d4;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #106ebe;
                border-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        self.expand_all_btn.clicked.connect(self.expand_all_blocks)
        
        # Add stretch to push buttons to the right
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.collapse_all_btn)
        toolbar_layout.addWidget(self.expand_all_btn)
        
        # Use QWebEngineView for advanced HTML/JavaScript support
        self.chat_display = QWebEngineView()
        # Note: QWebEnginePage doesn't have linkClicked signal in newer versions
        
        # Initialize with empty HTML content
        self.current_html_content = ""
        
        # Connect loadFinished signal to ensure page is ready
        self.chat_display.loadFinished.connect(self.on_page_load_finished)
        self.initialize_web_content()
        
        # Style the web engine view
        self.chat_display.setStyleSheet("""
            QWebEngineView {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 12px;
            }
        """)
        
        # Input area with modern styling
        input_layout = QHBoxLayout()
        input_layout.setSpacing(12)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about the patient database...")
        # Use system default font
        input_font = QApplication.font()
        input_font.setPointSize(14)
        self.input_field.setFont(input_font)
        self.input_field.setMinimumHeight(50)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 2px solid #3d3d3d;
                border-radius: 12px;
                padding: 15px 20px;
                font-family: Arial;
                font-size: 14px;
                font-weight: 400;
            }
            QLineEdit:focus {
                border-color: #0078d4;
                background-color: #333333;
            }
            QLineEdit::placeholder {
                color: #888888;
            }
        """)
        
        self.send_button = QPushButton("Send")
        # Use system default font
        button_font = QApplication.font()
        button_font.setPointSize(14)
        button_font.setWeight(QFont.Weight.DemiBold)
        self.send_button.setFont(button_font)
        self.send_button.setMinimumHeight(50)
        self.send_button.setMinimumWidth(100)
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0078d4, stop:1 #005a9e);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 15px 25px;
                font-family: Arial;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #106ebe, stop:1 #0f5794);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #005a9e, stop:1 #004578);
            }
            QPushButton:disabled {
                background: #444444;
                color: #888888;
            }
        """)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        
        # Add to main layout
        layout.addLayout(toolbar_layout)
        layout.addWidget(self.chat_display, 1)
        layout.addLayout(input_layout)
        
        self.setLayout(layout)
        
        # Track page load state
        self.page_loaded = False
    
    def on_page_load_finished(self, success):
        """Called when the web page finishes loading."""
        if success:
            self.page_loaded = True
    
    def initialize_web_content(self):
        """Initialize the web content with HTML structure and JavaScript."""
        initial_html = """
        <html>
        <head>
            <style>
            /* Table styling */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 16px 0;
                background: rgba(40, 44, 52, 0.8);
                border-radius: 8px;
                overflow: hidden;
            }
            th, td {
                border: 1px solid #444;
                padding: 12px 16px;
                text-align: left;
            }
            th {
                background-color: #2d3748;
                color: #e2e8f0;
                font-weight: bold;
            }
            td {
                background-color: rgba(45, 55, 72, 0.5);
            }
            tr:nth-child(even) td {
                background-color: rgba(45, 55, 72, 0.7);
            }
            tr:hover td {
                background-color: rgba(66, 153, 225, 0.2);
            }
            
            /* Code styling */
            code {
                background: #2d3748;
                padding: 2px 6px;
                border-radius: 4px;
                color: #a0aec0;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            }
            pre {
                background: #2d3748;
                padding: 16px;
                border-radius: 8px;
                overflow-x: auto;
                border: 1px solid #444;
            }
            
            /* General styling */
            h1, h2, h3, h4, h5, h6 {
                color: #e2e8f0;
                margin-top: 20px;
                margin-bottom: 10px;
            }
            p {
                line-height: 1.6;
                margin: 12px 0;
            }
            strong {
                color: #cbd5e0;
            }
            em {
                color: #a0aec0;
            }
            </style>
            <script>
            function toggleBlock(blockId) {
                var content = document.getElementById('content_' + blockId);
                var toggle = document.getElementById('toggle_' + blockId);
                
                if (content && toggle) {
                    if (content.style.display === 'none') {
                        content.style.display = 'block';
                        toggle.innerHTML = '▼ COLLAPSE';
                    } else {
                        content.style.display = 'none';
                        toggle.innerHTML = '▶ EXPAND';
                    }
                }
            }
            
            function collapseAllBlocks() {
                var allContentDivs = document.querySelectorAll('[id^="content_"]');
                var allToggleSpans = document.querySelectorAll('[id^="toggle_"]');
                
                allContentDivs.forEach(function(content) {
                    content.style.display = 'none';
                });
                
                allToggleSpans.forEach(function(toggle) {
                    toggle.innerHTML = '▶ EXPAND';
                });
            }
            
            function expandAllBlocks() {
                var allContentDivs = document.querySelectorAll('[id^="content_"]');
                var allToggleSpans = document.querySelectorAll('[id^="toggle_"]');
                
                allContentDivs.forEach(function(content) {
                    content.style.display = 'block';
                });
                
                allToggleSpans.forEach(function(toggle) {
                    toggle.innerHTML = '▼ COLLAPSE';
                });
            }
            
            function updateStreamingContent(messageId, content) {
                var contentDiv = document.getElementById(messageId + '_content');
                if (contentDiv) {
                    contentDiv.innerHTML = content;
                }
            }
            
            function addToolCallBlock(messageId, toolName, toolArgs, toolResult, collapsed) {
                var toolsContainer = document.getElementById(messageId + '_tools');
                if (!toolsContainer) {
                    // Create tools container if it doesn't exist
                    var parentDiv = document.getElementById(messageId);
                    if (parentDiv) {
                        var toolsDiv = document.createElement('div');
                        toolsDiv.id = messageId + '_tools';
                        toolsDiv.style.marginTop = '10px';
                        parentDiv.appendChild(toolsDiv);
                        toolsContainer = toolsDiv;
                    } else {
                        console.warn('Parent div not found for messageId: ' + messageId);
                        return;
                    }
                }
                
                if (toolsContainer) {
                    var toolId = messageId + '_tool_' + Date.now();
                    var displayStyle = collapsed ? 'none' : 'block';
                    var toggleText = collapsed ? '▶ EXPAND' : '▼ COLLAPSE';
                    
                    // Tool-specific colors and icons
                    var toolColor, toolIcon, toolBgColor;
                    if (toolName.includes('query_database')) {
                        toolColor = '#48bb78'; // Green for database queries
                        toolIcon = '🗃️';
                        toolBgColor = 'rgba(72, 187, 120, 0.1)';
                    } else if (toolName.includes('read_schema')) {
                        toolColor = '#4299e1'; // Blue for schema operations
                        toolIcon = '📋';
                        toolBgColor = 'rgba(66, 153, 225, 0.1)';
                    } else if (toolName.includes('insert') || toolName.includes('update')) {
                        toolColor = '#ed8936'; // Orange for modifications
                        toolIcon = '✏️';
                        toolBgColor = 'rgba(237, 137, 54, 0.1)';
                    } else if (toolName.includes('schedule')) {
                        toolColor = '#9f7aea'; // Purple for scheduling
                        toolIcon = '📅';
                        toolBgColor = 'rgba(159, 122, 234, 0.1)';
                    } else {
                        toolColor = '#a0aec0'; // Default gray
                        toolIcon = '🛠️';
                        toolBgColor = 'rgba(160, 174, 192, 0.1)';
                    }
                    
                    // Create tool div
                    var toolDiv = document.createElement('div');
                    toolDiv.style.cssText = 'margin: 8px 0; border: 2px solid ' + toolColor + '; border-radius: 8px; background: ' + toolBgColor + '; overflow: hidden;';
                    
                    // Create header
                    var headerDiv = document.createElement('div');
                    headerDiv.style.cssText = 'background: ' + toolColor + '22; padding: 8px 12px; border-bottom: 1px solid ' + toolColor + '44; cursor: pointer;';
                    headerDiv.onclick = function() { toggleBlock(toolId); };
                    headerDiv.innerHTML = '<table width="100%" style="border-collapse: collapse;"><tr>' +
                        '<td style="color: ' + toolColor + '; font-weight: bold; font-size: 12px;">' + toolIcon + ' ' + toolName + '</td>' +
                        '<td style="text-align: right; width: 80px;"><span id="toggle_' + toolId + '" style="color: ' + toolColor + '; font-size: 10px; background: ' + toolColor + '33; padding: 2px 6px; border-radius: 4px; cursor: pointer;">' + toggleText + '</span></td>' +
                        '</tr></table>';
                    
                    // Create content
                    var contentDiv = document.createElement('div');
                    contentDiv.id = 'content_' + toolId;
                    contentDiv.style.cssText = 'padding: 12px; background: ' + toolBgColor + '; display: ' + displayStyle + '; font-size: 12px;';
                    contentDiv.innerHTML = '<div style="margin-bottom: 8px;"><strong style="color: #ccc;">Arguments:</strong>' +
                        '<pre style="background: #2d3748; padding: 8px; border-radius: 4px; margin: 4px 0; overflow-x: auto; font-size: 11px; color: #e2e8f0;">' + toolArgs + '</pre></div>' +
                        '<div><strong style="color: #ccc;">Result:</strong>' +
                        '<pre style="background: #2d3748; padding: 8px; border-radius: 4px; margin: 4px 0; overflow-x: auto; font-size: 11px; color: #e2e8f0;">' + toolResult + '</pre></div>';
                    
                    toolDiv.appendChild(headerDiv);
                    toolDiv.appendChild(contentDiv);
                    
                    if (toolsContainer && toolsContainer.appendChild) {
                        toolsContainer.appendChild(toolDiv);
                    } else {
                        console.error('Unable to append tool div - toolsContainer is invalid');
                    }
                }
            }
            </script>
        </head>
        <body style='background-color: #1e1e1e; color: #ffffff; font-family: Arial; margin: 0; padding: 20px;'>
            <div id="messages"></div>
        </body>
        </html>
        """
        self.chat_display.setHtml(initial_html)
        
    def add_message(self, sender: str, message: str, is_user: bool = False):
        """Add a message to the chat display using JavaScript injection."""
        import time
        
        # Initialize tracking if needed
        if not hasattr(self, 'message_blocks'):
            self.message_blocks = {}
        
        # Generate unique block ID
        block_id = f"msg_{int(time.time() * 1000000)}"
        self.message_blocks[block_id] = {'expanded': True, 'type': 'message'}
        
        # Determine styling based on sender
        if is_user:
            border_color = "#00d4aa"
            bg_color = "rgba(0, 212, 170, 0.08)"
            icon = "USER"
            sender_display = "You"
        elif sender == "System":
            border_color = "#ff6b6b"
            bg_color = "rgba(255, 107, 107, 0.08)"
            icon = "SYS"
            sender_display = "System"
        else:
            border_color = "#0078d4"
            bg_color = "rgba(0, 120, 212, 0.08)"
            icon = "AI"
            sender_display = "AI Assistant"
        
        # Render markdown
        rendered_content = self.markdown_renderer.render_markdown(message)
        
        # Escape content for JavaScript
        escaped_content = rendered_content.replace("'", "\\'").replace("\n", "\\n").replace("\r", "")
        
        # Create message HTML using JavaScript
        if is_user:
            # Simple user message without collapse functionality
            js_code = f"""
            var messagesDiv = document.getElementById('messages');
            var messageDiv = document.createElement('div');
            messageDiv.innerHTML = `
                <div style="
                    margin: 15px 0;
                    border-left: 4px solid {border_color};
                    background: {bg_color};
                    border-radius: 8px;
                    padding: 16px 20px;
                ">
                    <div style="
                        color: {border_color};
                        font-weight: bold;
                        font-size: 14px;
                        margin-bottom: 8px;
                    ">
                        👤 {sender_display}
                    </div>
                    <div style="color: #e2e8f0;">
                        {escaped_content}
                    </div>
                </div>
            `;
            messagesDiv.appendChild(messageDiv);
            window.scrollTo(0, document.body.scrollHeight);
            """
        else:
            # AI/System messages with collapse functionality and tool name display
            tool_indicator = ""
            if sender != "System":
                # Check if this message has tool calls by looking for database-related content
                if any(keyword in message.lower() for keyword in ['patient', 'database', 'query', 'table', 'sql']):
                    tool_indicator = "🛠️ Used Database Tools • "
            
            js_code = f"""
            var messagesDiv = document.getElementById('messages');
            var messageDiv = document.createElement('div');
            messageDiv.innerHTML = `
                <div style="
                    margin: 15px 0;
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    background: {bg_color};
                    overflow: hidden;
                ">
                    <div style="
                        background: linear-gradient(135deg, {border_color}22, {border_color}11);
                        padding: 12px 20px;
                        border-bottom: 1px solid {border_color}44;
                        cursor: pointer;
                    " onclick="toggleBlock('{block_id}')">
                        <table width="100%" style="border-collapse: collapse;">
                            <tr>
                                <td style="color: {border_color}; font-weight: bold; font-size: 14px;">
                                    {icon} {tool_indicator}{sender_display}
                                </td>
                                <td style="text-align: right; width: 80px;">
                                    <span id="toggle_{block_id}" style="
                                        color: {border_color};
                                        font-size: 12px;
                                        background: {border_color}33;
                                        padding: 4px 8px;
                                        border-radius: 4px;
                                        cursor: pointer;
                                    ">▼ COLLAPSE</span>
                                </td>
                            </tr>
                        </table>
                    </div>
                    <div id="content_{block_id}" style="
                        padding: 20px;
                        background: {bg_color};
                        border-top: 1px solid {border_color}22;
                        display: block;
                    ">
                        {escaped_content}
                    </div>
                </div>
            `;
            messagesDiv.appendChild(messageDiv);
            window.scrollTo(0, document.body.scrollHeight);
            """
        
        # Execute JavaScript to add the message (with delay if page not loaded)
        if self.page_loaded:
            self.chat_display.page().runJavaScript(js_code)
        else:
            # Wait for page to load, then execute
            QTimer.singleShot(100, lambda: self.chat_display.page().runJavaScript(js_code))
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the chat area."""
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_tool_call_block(self, tool_name: str, tool_args: str, tool_result: str):
        """Add a tool call as HTML with collapsible dropdown."""
        import time
        
        # Initialize tracking if needed
        if not hasattr(self, 'message_blocks'):
            self.message_blocks = {}
        
        # Generate unique block IDs
        block_id = f"tool_{int(time.time() * 1000000)}"
        args_id = f"args_{block_id}"
        result_id = f"result_{block_id}"
        
        self.message_blocks[args_id] = {'expanded': False, 'type': 'tool_args'}
        self.message_blocks[result_id] = {'expanded': False, 'type': 'tool_result'}
        
        # Format tool arguments
        if tool_args and tool_args != "[args not available]": 
            try:
                import json
                args_dict = json.loads(tool_args)
                formatted_args = json.dumps(args_dict, indent=2)
            except:
                formatted_args = tool_args
        else:
            formatted_args = "No arguments"
        
        # Format tool results
        if tool_result:
            formatted_result = tool_result
        else:
            formatted_result = "No result"
        
        # Create HTML for tool call with dropdowns
        html_block = f"""
        <div style="
            margin: 15px 0;
            border: 2px solid #ffa500;
            border-radius: 12px;
            background: rgba(255, 165, 0, 0.08);
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(135deg, #ffa50022, #ffa50011);
                padding: 12px 20px;
                border-bottom: 1px solid #ffa50044;
            ">
                <div style="
                    color: #ffa500;
                    font-weight: bold;
                    font-size: 14px;
                    margin-bottom: 10px;
                ">
                    🔧 TOOL CALL: {tool_name}
                </div>
                
                <!-- Tool Arguments Section -->
                <div style="
                    background: #ffa50033;
                    border: 1px solid #ffa50055;
                    border-radius: 8px;
                    margin-bottom: 10px;
                    overflow: hidden;
                ">
                    <div style="
                        background: #ffa50044;
                        padding: 8px 15px;
                        cursor: pointer;
                        border-bottom: 1px solid #ffa50055;
                    " onclick="toggleBlock('{args_id}')">
                        <table width="100%" style="border-collapse: collapse;">
                            <tr>
                                <td style="color: #ffa500; font-weight: bold; font-size: 12px;">
                                    Arguments
                                </td>
                                <td style="text-align: right; width: 80px;">
                                    <span id="toggle_{args_id}" style="
                                        color: #ffa500;
                                        font-size: 11px;
                                        background: #ffa50033;
                                        padding: 3px 6px;
                                        border-radius: 3px;
                                        cursor: pointer;
                                    ">▶ EXPAND</span>
                                </td>
                            </tr>
                        </table>
                    </div>
                    <div id="content_{args_id}" style="
                        padding: 15px;
                        background: #ffa50011;
                        font-family: 'Courier New', monospace;
                        font-size: 12px;
                        white-space: pre-wrap;
                        display: none;
                    ">{formatted_args}</div>
                </div>
                
                <!-- Tool Results Section -->
                <div style="
                    background: #ffa50033;
                    border: 1px solid #ffa50055;
                    border-radius: 8px;
                    overflow: hidden;
                ">
                    <div style="
                        background: #ffa50044;
                        padding: 8px 15px;
                        cursor: pointer;
                        border-bottom: 1px solid #ffa50055;
                    " onclick="toggleBlock('{result_id}')">
                        <table width="100%" style="border-collapse: collapse;">
                            <tr>
                                <td style="color: #ffa500; font-weight: bold; font-size: 12px;">
                                    Results
                                </td>
                                <td style="text-align: right; width: 80px;">
                                    <span id="toggle_{result_id}" style="
                                        color: #ffa500;
                                        font-size: 11px;
                                        background: #ffa50033;
                                        padding: 3px 6px;
                                        border-radius: 3px;
                                        cursor: pointer;
                                    ">▶ EXPAND</span>
                                </td>
                            </tr>
                        </table>
                    </div>
                    <div id="content_{result_id}" style="
                        padding: 15px;
                        background: #ffa50011;
                        font-family: 'Courier New', monospace;
                        font-size: 12px;
                        white-space: pre-wrap;
                        display: none;
                    ">{formatted_result}</div>
                </div>
            </div>
        </div>
        """
        
        # Insert HTML into chat display
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.insertHtml(html_block)
        
        # Auto-scroll to bottom
        self.chat_display.ensureCursorVisible()
    
    # Old HTML-based methods removed - now using widget-based approach
    
    def _cleaned_placeholder(self):
        """Add a message as a distinct, collapsible block."""
        import time
        
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        
        # Add spacing between blocks
        if cursor.position() > 0:
            self.chat_display.insertHtml("<br>")
        
        # Generate unique block ID
        block_id = f"{block_type}_{int(time.time() * 1000000)}"
        
        # Initialize message blocks tracking
        if not hasattr(self, 'message_blocks'):
            self.message_blocks = {}
        self.message_blocks[block_id] = {'expanded': True, 'type': block_type}
        
        # Determine block styling based on sender
        if is_user:
            border_color = "#00d4aa"
            bg_color = "rgba(0, 212, 170, 0.05)"
            icon = "USER"
            sender_name = "You"
        elif sender == "System":
            border_color = "#ff6b6b"
            bg_color = "rgba(255, 107, 107, 0.05)"
            icon = "SYS"
            sender_name = "System"
        else:
            border_color = "#0078d4"
            bg_color = "rgba(0, 120, 212, 0.05)"
            icon = "AI"
            sender_name = "Assistant"
        
        # Render message content as markdown
        message_html = self.markdown_renderer.render_markdown(message)
        
        # Create collapsible message block
        block_html = f'''
        <div id="{block_id}" style="
            display: block; 
            margin: 12px 0; 
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: 8px;
            clear: both;
        ">
            <!-- Collapsible Header -->
            <div style="
                display: block;
                padding: 12px 15px;
                background-color: {bg_color};
                border-bottom: 1px solid {border_color};
                cursor: pointer;
                border-radius: 7px 7px 0 0;
            " data-message-block-id="{block_id}">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="vertical-align: middle;">
                            <strong style="color: {border_color}; font-size: 14px; font-weight: 600;">{icon} {sender_name}</strong>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">
                            <span style="
                                background-color: {border_color}; 
                                color: white; 
                                padding: 6px 12px; 
                                border-radius: 15px; 
                                font-size: 11px; 
                                font-weight: bold;
                                cursor: pointer;
                                display: inline-block;
                                min-width: 70px;
                                text-align: center;
                                border: 2px solid {border_color};
                            " id="{block_id}_arrow">COLLAPSE</span>
                        </td>
                    </tr>
                </table>
            </div>
            
            <!-- Message Content -->
            <div id="{block_id}_content" style="
                display: block;
                padding: 15px;
                border-radius: 0 0 7px 7px;
            ">
                {message_html}
            </div>
        </div>
        '''
        
        self.chat_display.insertHtml(block_html)
        self.chat_display.ensureCursorVisible()
    
    def toggle_message_block(self, block_id: str):
        """Toggle the expansion state of a message block."""
        if not hasattr(self, 'message_blocks'):
            self.message_blocks = {}
        
        # If block not found, initialize it as expanded
        if block_id not in self.message_blocks:
            self.message_blocks[block_id] = {'expanded': True, 'type': 'message'}
        
        block_info = self.message_blocks[block_id]
        is_expanded = block_info.get('expanded', True)
        
        # Store scroll position
        scrollbar = self.chat_display.verticalScrollBar()
        scroll_position = scrollbar.value()
        
        # Get current HTML
        current_html = self.chat_display.toHtml()
        
        if is_expanded:
            # Collapse: hide content and change button
            updated_html = current_html.replace(
                f'id="{block_id}_content" style="display: block;',
                f'id="{block_id}_content" style="display: none;'
            ).replace(
                f'id="{block_id}_content" style="\n                display: block;',
                f'id="{block_id}_content" style="\n                display: none;'
            ).replace(
                f'id="{block_id}_arrow">COLLAPSE</span>',
                f'id="{block_id}_arrow">EXPAND</span>'
            )
            block_info['expanded'] = False
        else:
            # Expand: show content and change button
            updated_html = current_html.replace(
                f'id="{block_id}_content" style="display: none;',
                f'id="{block_id}_content" style="display: block;'
            ).replace(
                f'id="{block_id}_content" style="\n                display: none;',
                f'id="{block_id}_content" style="\n                display: block;'
            ).replace(
                f'id="{block_id}_arrow">EXPAND</span>',
                f'id="{block_id}_arrow">COLLAPSE</span>'
            )
            block_info['expanded'] = True
        
        # Update the display
        if updated_html != current_html:
            self.chat_display.setHtml(updated_html)
            # Restore scroll position
            scrollbar.setValue(scroll_position)
    
    def _add_tool_call_block(self, tool_name: str, tool_args: str, tool_result: str):
        """Add a tool call as its own collapsible block."""
        import time
        
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        
        # Add spacing
        if cursor.position() > 0:
            self.chat_display.insertHtml("<br>")
        
        # Generate unique block ID
        block_id = f"tool_call_{int(time.time() * 1000000)}"
        
        # Format tool arguments and results
        try:
            import json
            if tool_args.startswith('{') or tool_args.startswith('['):
                formatted_args = json.dumps(json.loads(tool_args), indent=2)
            else:
                formatted_args = tool_args
                
            if tool_result.startswith('{') or tool_result.startswith('['):
                formatted_result = json.dumps(json.loads(tool_result), indent=2)
            else:
                formatted_result = tool_result
        except:
            formatted_args = tool_args
            formatted_result = tool_result
        
        # Escape HTML
        import html
        escaped_args = html.escape(formatted_args)
        escaped_result = html.escape(formatted_result)
        
        # Determine status and color
        status_icon = "[OK]" if "success" in tool_result.lower() else "[ERR]" if "error" in tool_result.lower() else "[INFO]"
        status_color = "#4caf50" if "success" in tool_result.lower() else "#ff9800" if "error" in tool_result.lower() else "#2196f3"
        
        # Store block ID for toggle functionality
        if not hasattr(self, 'tool_call_blocks'):
            self.tool_call_blocks = {}
        self.tool_call_blocks[block_id] = {'expanded': True}
        
        # Create collapsible tool call block with consistent design
        tool_block_html = f'''
        <div id="{block_id}" style="
            display: block; 
            margin: 12px 0; 
            background-color: rgba(76, 175, 80, 0.05);
            border: 1px solid {status_color};
            border-radius: 8px;
            clear: both;
        ">
            <!-- Collapsible Header -->
            <div style="
                display: block;
                padding: 12px 15px;
                background-color: rgba(76, 175, 80, 0.05);
                border-bottom: 1px solid {status_color};
                cursor: pointer;
                border-radius: 7px 7px 0 0;
            " data-tool-block-id="{block_id}">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="vertical-align: middle;">
                            <strong style="color: {status_color}; font-size: 14px; font-weight: 600;">TOOL: {tool_name} {status_icon}</strong>
                        </td>
                        <td style="text-align: right; vertical-align: middle;">
                            <span style="
                                background-color: {status_color}; 
                                color: white; 
                                padding: 6px 12px; 
                                border-radius: 15px; 
                                font-size: 11px; 
                                font-weight: bold;
                                cursor: pointer;
                                display: inline-block;
                                min-width: 70px;
                                text-align: center;
                                border: 2px solid {status_color};
                            " id="{block_id}_arrow">COLLAPSE</span>
                        </td>
                    </tr>
                </table>
            </div>
            
            <!-- Tool Content -->
            <div id="{block_id}_content" style="
                display: block;
                padding: 15px;
                border-radius: 0 0 7px 7px;
            ">
                <!-- Tool Arguments -->
                <div style="margin-bottom: 15px;">
                    <div style="font-size: 12px; color: #888; margin-bottom: 8px; font-weight: 600;">Arguments:</div>
                    <div style="
                        padding: 10px; 
                        background-color: #2a2a2a; 
                        border-radius: 4px;
                        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;\n            font-size: 13px;
                        font-size: 11px;
                        color: #ffffff;
                        white-space: pre-wrap;
                        max-height: 150px;
                        overflow-y: auto;
                        border: 1px solid #444;
                    ">{escaped_args}</div>
                </div>
                
                <!-- Tool Results -->
                <div>
                    <div style="font-size: 12px; color: #888; margin-bottom: 8px; font-weight: 600;">Result:</div>
                    <div style="
                        padding: 10px; 
                        background-color: #2a2a2a; 
                        border-radius: 4px;
                        font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;\n            font-size: 13px;
                        font-size: 11px;
                        color: #ffffff;
                        white-space: pre-wrap;
                        max-height: 300px;
                        overflow-y: auto;
                        border: 1px solid #444;
                    ">{escaped_result}</div>
                </div>
            </div>
        </div>
        '''
        
        self.chat_display.insertHtml(tool_block_html)
        self.chat_display.ensureCursorVisible()
    
    def toggle_tool_call_block(self, block_id: str):
        """Toggle the expansion state of a tool call block using JavaScript-like approach."""
        if not hasattr(self, 'tool_call_blocks'):
            self.tool_call_blocks = {}
        
        # If block not found, initialize it as expanded
        if block_id not in self.tool_call_blocks:
            self.tool_call_blocks[block_id] = {'expanded': True}
        
        block_info = self.tool_call_blocks[block_id]
        is_expanded = block_info.get('expanded', True)
        
        # Store scroll position
        scrollbar = self.chat_display.verticalScrollBar()
        scroll_position = scrollbar.value()
        
        # Get current HTML
        current_html = self.chat_display.toHtml()
        
        if is_expanded:
            # Collapse: hide content and change arrow
            updated_html = current_html.replace(
                f'id="{block_id}_content" style="display: block;',
                f'id="{block_id}_content" style="display: none;'
            ).replace(
                f'id="{block_id}_content" style="\n                display: block;',
                f'id="{block_id}_content" style="\n                display: none;'
            ).replace(
                '>COLLAPSE</span>',
                '>EXPAND</span>'
            )
            block_info['expanded'] = False
        else:
            # Expand: show content and change arrow
            updated_html = current_html.replace(
                f'id="{block_id}_content" style="display: none;',
                f'id="{block_id}_content" style="display: block;'
            ).replace(
                f'id="{block_id}_content" style="\n                display: none;',
                f'id="{block_id}_content" style="\n                display: block;'
            ).replace(
                '>EXPAND</span>',
                '>COLLAPSE</span>'
            )
            block_info['expanded'] = True
        
        # Update the display
        self.chat_display.setHtml(updated_html)
        
        # Restore scroll position
        scrollbar.setValue(scroll_position)
        
    def add_thinking_message(self, message: str):
        """Add a thinking message as a collapsible block."""
        # For now, add thinking messages as regular AI messages
        self.add_message("AI (Thinking)", message, is_user=False)
    
    def toggle_thinking_block(self, block_id: str):
        """Toggle the expansion state of a thinking block."""
        if not hasattr(self, 'thinking_blocks'):
            self.thinking_blocks = {}
        
        # If block not found, initialize it as collapsed
        if block_id not in self.thinking_blocks:
            self.thinking_blocks[block_id] = {'expanded': False}
        
        block_info = self.thinking_blocks[block_id]
        is_expanded = block_info.get('expanded', False)
        
        # Store scroll position
        scrollbar = self.chat_display.verticalScrollBar()
        scroll_position = scrollbar.value()
        
        # Get current HTML
        current_html = self.chat_display.toHtml()
        
        if is_expanded:
            # Collapse: hide content and change arrow
            updated_html = current_html.replace(
                f'id="{block_id}_content" style="display: block;',
                f'id="{block_id}_content" style="display: none;'
            ).replace(
                f'id="{block_id}_content" style="\n                display: block;',
                f'id="{block_id}_content" style="\n                display: none;'
            ).replace(
                '>COLLAPSE</span>',
                '>EXPAND</span>'
            )
            block_info['expanded'] = False
        else:
            # Expand: show content and change arrow
            updated_html = current_html.replace(
                f'id="{block_id}_content" style="display: none;',
                f'id="{block_id}_content" style="display: block;'
            ).replace(
                f'id="{block_id}_content" style="\n                display: none;',
                f'id="{block_id}_content" style="\n                display: block;'
            ).replace(
                '>EXPAND</span>',
                '>COLLAPSE</span>'
            )
            block_info['expanded'] = True
        
        # Update the display
        if updated_html != current_html:
            self.chat_display.setHtml(updated_html)
            # Restore scroll position
            scrollbar.setValue(scroll_position)
    
    def collapse_all_blocks(self):
        """Collapse all blocks in the chat."""
        # Collapse all message blocks
        if hasattr(self, 'message_blocks'):
            for block_id in self.message_blocks:
                if self.message_blocks[block_id].get('expanded', True):
                    self.toggle_message_block(block_id)
        
        # Collapse all thinking blocks
        if hasattr(self, 'thinking_blocks'):
            for block_id in self.thinking_blocks:
                if self.thinking_blocks[block_id].get('expanded', False):
                    self.toggle_thinking_block(block_id)
        
        # Collapse all tool call blocks
        if hasattr(self, 'tool_call_blocks'):
            for block_id in self.tool_call_blocks:
                if self.tool_call_blocks[block_id].get('expanded', True):
                    self.toggle_tool_call_block(block_id)
    
    def expand_all_blocks(self):
        """Expand all blocks in the chat."""
        # Expand all message blocks
        if hasattr(self, 'message_blocks'):
            for block_id in self.message_blocks:
                if not self.message_blocks[block_id].get('expanded', True):
                    self.toggle_message_block(block_id)
        
        # Expand all thinking blocks
        if hasattr(self, 'thinking_blocks'):
            for block_id in self.thinking_blocks:
                if not self.thinking_blocks[block_id].get('expanded', False):
                    self.toggle_thinking_block(block_id)
        
        # Expand all tool call blocks
        if hasattr(self, 'tool_call_blocks'):
            for block_id in self.tool_call_blocks:
                if not self.tool_call_blocks[block_id].get('expanded', True):
                    self.toggle_tool_call_block(block_id)
    
    def collapse_all_blocks(self):
        """Collapse all foldable blocks in the chat using JavaScript."""
        if self.page_loaded:
            self.chat_display.page().runJavaScript("collapseAllBlocks();")
    
    def expand_all_blocks(self):
        """Expand all foldable blocks in the chat using JavaScript."""
        if self.page_loaded:
            self.chat_display.page().runJavaScript("expandAllBlocks();")
    
    def start_streaming_response(self):
        """Start a streaming response placeholder."""
        import time
        
        # Generate unique ID for streaming message
        self.streaming_message_id = f"streaming_{int(time.time() * 1000000)}"
        
        # Create streaming placeholder using JavaScript
        js_code = f"""
        var messagesDiv = document.getElementById('messages');
        var streamingDiv = document.createElement('div');
        streamingDiv.innerHTML = `
            <div id="{self.streaming_message_id}" style="
                margin: 15px 0;
                border: 2px solid #0078d4;
                border-radius: 12px;
                background: rgba(0, 120, 212, 0.08);
                overflow: hidden;
            ">
                <div style="
                    background: linear-gradient(135deg, #0078d422, #0078d411);
                    padding: 12px 20px;
                    border-bottom: 1px solid #0078d444;
                ">
                    <div style="color: #0078d4; font-weight: bold; font-size: 14px;">
                        AI Assistant (streaming...)
                    </div>
                </div>
                <div id="{self.streaming_message_id}_content" style="
                    padding: 20px;
                    background: rgba(0, 120, 212, 0.08);
                    min-height: 30px;
                ">
                    <span style="color: #888888;">Generating response...</span>
                </div>
                <div id="{self.streaming_message_id}_tools" style="
                    margin-top: 10px;
                ">
                </div>
            </div>
        `;
        messagesDiv.appendChild(streamingDiv);
        window.scrollTo(0, document.body.scrollHeight);
        """
        
        # Execute JavaScript to add streaming placeholder
        if self.page_loaded:
            self.chat_display.page().runJavaScript(js_code)
        else:
            QTimer.singleShot(100, lambda: self.chat_display.page().runJavaScript(js_code))
    
    def _filter_tool_results_from_text(self, text: str) -> str:
        """Filter out tool call results from streaming text."""
        import re
        
        # Patterns to match various tool result formats
        patterns_to_remove = [
            # JSON tool results
            r'\{[\'"]status[\'"]:\s*[\'"]success[\'"][^}]*\}',
            # Multiple consecutive identical JSON objects
            r'(\{[^}]*\})\1+',
            # Tool call results in brackets
            r'\[.*?tool.*?result.*?\]',
            # Database query results
            r'\{[\'"]query_type[\'"]:[^}]*\}',
            # Status results
            r'\{[\'"]status[\'"]:[^}]*\}',
            # Repeated closing brackets/braces (common in tool results)
            r'[\]\}]{3,}',
            # Repeated opening brackets/braces
            r'[\[\{]{3,}',
            # JSON fragments at start of text
            r'^[\]\}]+',
            # Tool result patterns
            r'\{[\'"]results[\'"]:[^}]*\}',
            r'\{[\'"]columns[\'"]:[^}]*\}',
            r'\{[\'"]row_count[\'"]:[^}]*\}',
        ]
        
        filtered_text = text
        
        # Apply all patterns
        for pattern in patterns_to_remove:
            filtered_text = re.sub(pattern, '', filtered_text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove repetitive patterns and clean up lines
        lines = filtered_text.split('\n')
        filtered_lines = []
        prev_line = ""
        consecutive_json_count = 0
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Detect JSON-like tool results
            is_tool_result = (
                (line.startswith('{') and line.endswith('}') and 'status' in line) or
                (line.startswith('{') and line.endswith('}') and 'query_type' in line) or
                (line.startswith('{') and line.endswith('}') and 'results' in line)
            )
            
            # Skip tool results
            if is_tool_result:
                consecutive_json_count += 1
                continue
            else:
                consecutive_json_count = 0
            
            # Skip identical consecutive lines (repetitive tool results)
            if line != prev_line:
                # Additional check for tool-like content
                if not (line.count('{') > 2 and line.count('}') > 2):
                    filtered_lines.append(line)
                    prev_line = line
        
        # Join back and clean up extra whitespace
        result = '\n'.join(filtered_lines).strip()
        
        # Remove multiple consecutive newlines
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        # Remove leading/trailing JSON artifacts
        result = re.sub(r'^[\{\}]+|[\{\}]+$', '', result).strip()
        
        return result
    
    def update_streaming_response(self, partial_text: str):
        """Update the streaming response with new content."""
        if hasattr(self, 'streaming_message_id') and self.streaming_message_id:
            # Filter out tool results from the main response
            filtered_text = self._filter_tool_results_from_text(partial_text)
            
            # Escape HTML in the partial text and render as markdown
            rendered_content = self.markdown_renderer.render_markdown(filtered_text)
            
            # Update the content using JavaScript
            escaped_content = rendered_content.replace("'", "\\'").replace("\n", "\\n").replace("\r", "")
            js_code = f"""
            updateStreamingContent('{self.streaming_message_id}', '{escaped_content}');
            window.scrollTo(0, document.body.scrollHeight);
            """
            if self.page_loaded:
                self.chat_display.page().runJavaScript(js_code)
    
    def add_tool_call_to_stream(self, tool_name: str, tool_args: str, tool_result: str, collapsed: bool = True):
        """Add a tool call block to the current streaming message."""
        if hasattr(self, 'streaming_message_id') and self.streaming_message_id and self.page_loaded:
            # Escape content for JavaScript
            escaped_args = tool_args.replace("'", "\\'").replace("\n", "\\n").replace("\r", "").replace("`", "\\`")
            escaped_result = tool_result.replace("'", "\\'").replace("\n", "\\n").replace("\r", "").replace("`", "\\`")
            escaped_name = tool_name.replace("'", "\\'")
            
            js_code = f"""
            addToolCallBlock('{self.streaming_message_id}', '{escaped_name}', '{escaped_args}', '{escaped_result}', {str(collapsed).lower()});
            window.scrollTo(0, document.body.scrollHeight);
            """
            self.chat_display.page().runJavaScript(js_code)
    
    def finish_streaming_response(self):
        """Finish the streaming response and clean up."""
        if hasattr(self, 'streaming_message_id') and self.streaming_message_id and self.page_loaded:
            # Update the header to remove "streaming..." indicator
            js_code = f"""
            var headerDiv = document.querySelector('#{self.streaming_message_id} div div');
            if (headerDiv && headerDiv.textContent.includes('streaming...')) {{
                headerDiv.textContent = 'AI Assistant';
            }}
            """
            self.chat_display.page().runJavaScript(js_code)
            
            # Clear streaming ID
            self.streaming_message_id = None
    
    def handle_anchor_click(self, url):
        """Handle clicks on anchor links in the chat display."""
        # This is kept for compatibility but we now use mouse events
        pass


class PatientDatabaseChatGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.agent = None
        self.agent_worker = None
        self.agent_thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.init_ui()
        self.init_agent()
        self.show_welcome_message()
        
    def init_ui(self):
        """Initialize the user interface with modern design."""
        self.setWindowTitle("Patient Database Chat Assistant")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 600)
        
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)
        
        # Add line break before tool result
        if cursor.position() > 0:
            self.chat_display.insertHtml("<br>")
        
        # Generate unique ID for this tool result
        tool_result_id = f"tool_result_{len(self.tool_details)}"
        
        # Format JSON for display
        formatted_result = tool_result
        try:
            import json
            if tool_result.strip().startswith('{') or tool_result.strip().startswith('['):
                parsed = json.loads(tool_result)
                formatted_result = json.dumps(parsed, indent=2)
        except:
            pass
        
        # Store detailed tool information
        self.tool_details[tool_result_id] = {
            'name': tool_name,
            'args': tool_args,
            'result': tool_result,
            'formatted_result': formatted_result,
            'expanded': False,
            'button': None,  # Will store the button reference
            'content_id': f"{tool_result_id}_content"
        }
        
        # Create initial collapsed tool result using the same generator function
        result_html = self._generate_tool_result_html(tool_result_id, False)
        
        self.chat_display.insertHtml(result_html)
        self.chat_display.ensureCursorVisible()
        
        # Clear stored tool call
        self.last_tool_call = None
    
    def toggle_tool_result_simple(self, tool_result_id: str):
        """Toggle tool result expansion in place using HTML replacement."""
        if tool_result_id not in self.tool_details:
            return
        
        tool_info = self.tool_details[tool_result_id]
        is_expanded = tool_info.get('expanded', False)
        
        # Store scroll position to prevent page jumping
        scrollbar = self.chat_display.verticalScrollBar()
        scroll_position = scrollbar.value()
        
        # Get current HTML content
        current_html = self.chat_display.toHtml()
        
        # Generate the new HTML for this tool result
        new_tool_html = self._generate_tool_result_html(tool_result_id, not is_expanded)
        
        # Find and replace the current tool result HTML
        updated_html = self._replace_tool_result_in_html(current_html, tool_result_id, new_tool_html)
        
        if updated_html != current_html:
            # Update the display with new HTML
            self.chat_display.setHtml(updated_html)
            
            # Update the expanded state
            tool_info['expanded'] = not is_expanded
            
            # Restore scroll position
            scrollbar.setValue(scroll_position)
    
    def _generate_tool_result_html(self, tool_result_id: str, expanded: bool) -> str:
        """Generate HTML for a tool result in collapsed or expanded state."""
        tool_info = self.tool_details[tool_result_id]
        tool_name = tool_info['name']
        formatted_result = tool_info['formatted_result']
        
        # Determine colors based on result content
        status_icon = "[OK]" if "success" in formatted_result.lower() else "[ERR]" if "error" in formatted_result.lower() else "[INFO]"
        status_color = "#4caf50" if "success" in formatted_result.lower() else "#ff9800" if "error" in formatted_result.lower() else "#2196f3"
        
        # Get result summary
        result_summary = ""
        if "status" in formatted_result.lower() and "success" in formatted_result.lower():
            result_summary = "successful"
        elif "count" in formatted_result.lower() or "row_count" in formatted_result.lower():
            try:
                import json
                if formatted_result.strip().startswith('{'):
                    data = json.loads(formatted_result)
                    if 'row_count' in data:
                        result_summary = f"{data['row_count']} rows"
                    elif 'results' in data and isinstance(data['results'], list):
                        result_summary = f"{len(data['results'])} results"
            except:
                pass
        
        if expanded:
            # Expanded state with content
            import html
            escaped_result = html.escape(formatted_result)
            
            arrow = "▼"
            action_text = "[click to collapse]"
            button_text = f"{arrow} {status_icon} {tool_name}"
            if result_summary:
                button_text += f" • {result_summary}"
            button_text += f" {action_text}"
            
            return f'''<div style="display: block; margin: 8px 0; clear: both;" id="{tool_result_id}_container">
    <div style="display: block; padding: 8px 12px; background-color: rgba(76, 175, 80, 0.1); border-left: 3px solid {status_color}; border-radius: 4px; cursor: pointer; user-select: none; font-size: 12px; font-weight: 600; color: {status_color};" data-tool-id="{tool_result_id}">
        {button_text}
    </div>
    <div style="margin: 4px 0 8px 20px; padding: 10px; background-color: #2a2a2a; border: 1px solid #555555; border-radius: 4px; font-family: monospace; font-size: 11px; color: #ffffff; white-space: pre-wrap; max-height: 300px; overflow-y: auto;">
        <div style="color: #00d4aa; font-size: 10px; margin-bottom: 8px; font-weight: 600;">[RESULT] {tool_name}:</div>
        {escaped_result}
    </div>
</div>'''
        else:
            # Collapsed state
            arrow = "▶"
            action_text = "[click to expand]"
            button_text = f"{arrow} {status_icon} {tool_name}"
            if result_summary:
                button_text += f" • {result_summary}"
            button_text += f" {action_text}"
            
            return f'''<div style="display: block; margin: 8px 0; clear: both;" id="{tool_result_id}_container">
    <div style="display: block; padding: 8px 12px; background-color: rgba(76, 175, 80, 0.1); border-left: 3px solid {status_color}; border-radius: 4px; cursor: pointer; user-select: none; font-size: 12px; font-weight: 600; color: {status_color};" data-tool-id="{tool_result_id}">
        {button_text}
    </div>
</div>'''
    
    def _replace_tool_result_in_html(self, html_content: str, tool_result_id: str, new_html: str) -> str:
        """Replace a specific tool result in the HTML content."""
        import re
        
        # More precise pattern to find the complete container div
        # Look for the container div and count nested divs to find the proper closing tag
        def find_container_div(content, container_id):
            """Find the complete container div including all nested content."""
            start_pattern = rf'<div[^>]*id="{re.escape(container_id)}_container"[^>]*>'
            
            start_match = re.search(start_pattern, content)
            if not start_match:
                return None, None
            
            start_pos = start_match.start()
            pos = start_match.end()
            div_count = 1
            
            # Count nested divs to find the matching closing tag
            while pos < len(content) and div_count > 0:
                next_open = content.find('<div', pos)
                next_close = content.find('</div>', pos)
                
                if next_close == -1:
                    break
                
                if next_open != -1 and next_open < next_close:
                    div_count += 1
                    pos = next_open + 4
                else:
                    div_count -= 1
                    if div_count == 0:
                        end_pos = next_close + 6
                        return start_pos, end_pos
                    pos = next_close + 6
            
            return None, None
        
        # Find the container div boundaries
        start_pos, end_pos = find_container_div(html_content, tool_result_id)
        
        if start_pos is not None and end_pos is not None:
            # Replace the found container
            updated_html = html_content[:start_pos] + new_html + html_content[end_pos:]
            return updated_html
        else:
            # Fallback: try to find by data-tool-id attribute
            pattern = rf'<div[^>]*data-tool-id="{re.escape(tool_result_id)}"[^>]*>.*?</div>'
            match = re.search(pattern, html_content, flags=re.DOTALL)
            if match:
                # Find the parent container div
                before_match = html_content[:match.start()]
                last_div_start = before_match.rfind('<div')
                if last_div_start != -1:
                    # Find the closing div for this container
                    after_match = html_content[match.end():]
                    next_div_close = after_match.find('</div>')
                    if next_div_close != -1:
                        full_start = last_div_start
                        full_end = match.end() + next_div_close + 6
                        updated_html = html_content[:full_start] + new_html + html_content[full_end:]
                        return updated_html
        
        # If no replacement was made, return original
        return html_content
    
    def handle_anchor_click(self, url):
        """Handle clicks on anchor links in the chat display."""
        # This is kept for compatibility but we now use mouse events
        pass
    
    def update_tool_result_html(self, html_content: str, tool_result_id: str) -> str:
        """Update the HTML content to show the correct state for a tool result."""
        tool_info = self.tool_details[tool_result_id]
        is_expanded = tool_info.get('expanded', False)
        formatted_result = tool_info['formatted_result']
        tool_name = tool_info['name']
        
        # Determine colors and summary
        status_icon = "[OK]" if "success" in formatted_result.lower() else "[ERR]" if "error" in formatted_result.lower() else "[INFO]"
        status_color = "#4caf50" if "success" in formatted_result.lower() else "#ff9800" if "error" in formatted_result.lower() else "#2196f3"
        
        result_summary = ""
        if "status" in formatted_result.lower() and "success" in formatted_result.lower():
            result_summary = "successful"
        elif "count" in formatted_result.lower() or "row_count" in formatted_result.lower():
            try:
                import json
                if formatted_result.strip().startswith('{'):
                    data = json.loads(formatted_result)
                    if 'row_count' in data:
                        result_summary = f"{data['row_count']} rows"
                    elif 'results' in data and isinstance(data['results'], list):
                        result_summary = f"{len(data['results'])} results"
            except:
                pass
        
        # Create the new HTML based on expanded state
        if is_expanded:
            # Escape HTML in the formatted result
            import html
            escaped_result = html.escape(formatted_result)
            
            new_tool_html = f'''<!-- TOOL_RESULT_START_{tool_result_id} -->
        <div style="display: block; margin: 8px 0; clear: both;">
            <a href="http://toggle_tool/{tool_result_id}" style="text-decoration: none; color: inherit;">
                <div style="
                    display: block; 
                    padding: 6px 10px; 
                    background-color: rgba(76, 175, 80, 0.1); 
                    border-left: 3px solid {status_color}; 
                    border-radius: 4px; 
                    cursor: pointer;
                    user-select: none;
                " id="{tool_result_id}_header">
                    <span id="{tool_result_id}_arrow" style="color: {status_color}; font-size: 12px; font-weight: 600;">▼</span>
                    <span style="color: {status_color}; font-size: 12px; font-weight: 600; margin-left: 4px;">{status_icon} {tool_name}</span>
                    {f'<span style="color: #888888; font-size: 11px; margin-left: 8px;">{result_summary}</span>' if result_summary else ''}
                    <span style="color: #666666; font-size: 10px; margin-left: 8px; font-style: italic;">[click to collapse]</span>
                </div>
            </a>
            <div style="
                display: block; 
                margin-top: 4px; 
                padding: 10px; 
                background-color: #2a2a2a; 
                border: 1px solid #555555; 
                border-radius: 4px;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;\n            font-size: 13px;
                font-size: 11px;
                color: #ffffff;
                white-space: pre-wrap;
                max-height: 300px;
                overflow-y: auto;
            ">{escaped_result}</div>
        </div>
        <!-- TOOL_RESULT_END_{tool_result_id} -->'''
        else:
            new_tool_html = f'''<!-- TOOL_RESULT_START_{tool_result_id} -->
        <div style="display: block; margin: 8px 0; clear: both;">
            <a href="http://toggle_tool/{tool_result_id}" style="text-decoration: none; color: inherit;">
                <div style="
                    display: block; 
                    padding: 6px 10px; 
                    background-color: rgba(76, 175, 80, 0.1); 
                    border-left: 3px solid {status_color}; 
                    border-radius: 4px; 
                    cursor: pointer;
                    user-select: none;
                " id="{tool_result_id}_header">
                    <span id="{tool_result_id}_arrow" style="color: {status_color}; font-size: 12px; font-weight: 600;">▶</span>
                    <span style="color: {status_color}; font-size: 12px; font-weight: 600; margin-left: 4px;">{status_icon} {tool_name}</span>
                    {f'<span style="color: #888888; font-size: 11px; margin-left: 8px;">{result_summary}</span>' if result_summary else ''}
                    <span style="color: #666666; font-size: 10px; margin-left: 8px; font-style: italic;">[click to expand]</span>
                </div>
            </a>
        </div>
        <!-- TOOL_RESULT_END_{tool_result_id} -->'''
        
        # Find and replace using unique comment markers
        import re
        pattern = rf'<!-- TOOL_RESULT_START_{re.escape(tool_result_id)} -->.*?<!-- TOOL_RESULT_END_{re.escape(tool_result_id)} -->'
        
        # Replace with new HTML
        updated_html = re.sub(pattern, new_tool_html, html_content, flags=re.DOTALL)
        
        return updated_html
    
    def update_tool_result_html_by_id(self, html_content: str, tool_result_id: str) -> str:
        """Update the HTML content using div ID instead of comments."""
        tool_info = self.tool_details[tool_result_id]
        is_expanded = tool_info.get('expanded', False)
        formatted_result = tool_info['formatted_result']
        tool_name = tool_info['name']
        
        # Determine colors and summary
        status_icon = "[OK]" if "success" in formatted_result.lower() else "[ERR]" if "error" in formatted_result.lower() else "[INFO]"
        status_color = "#4caf50" if "success" in formatted_result.lower() else "#ff9800" if "error" in formatted_result.lower() else "#2196f3"
        
        result_summary = ""
        if "status" in formatted_result.lower() and "success" in formatted_result.lower():
            result_summary = "successful"
        elif "count" in formatted_result.lower() or "row_count" in formatted_result.lower():
            try:
                import json
                if formatted_result.strip().startswith('{'):
                    data = json.loads(formatted_result)
                    if 'row_count' in data:
                        result_summary = f"{data['row_count']} rows"
                    elif 'results' in data and isinstance(data['results'], list):
                        result_summary = f"{len(data['results'])} results"
            except:
                pass
        
        # Create the new HTML based on expanded state
        if is_expanded:
            # Escape HTML in the formatted result
            import html
            escaped_result = html.escape(formatted_result)
            
            new_tool_html = f'''<div style="display: block; margin: 8px 0; clear: both;" data-tool-id="{tool_result_id}">
            <a href="http://toggle_tool/{tool_result_id}" style="text-decoration: none; color: inherit;">
                <div style="
                    display: block; 
                    padding: 6px 10px; 
                    background-color: rgba(76, 175, 80, 0.1); 
                    border-left: 3px solid {status_color}; 
                    border-radius: 4px; 
                    cursor: pointer;
                    user-select: none;
                " id="{tool_result_id}_header">
                    <span id="{tool_result_id}_arrow" style="color: {status_color}; font-size: 12px; font-weight: 600;">▼</span>
                    <span style="color: {status_color}; font-size: 12px; font-weight: 600; margin-left: 4px;">{status_icon} {tool_name}</span>
                    {f'<span style="color: #888888; font-size: 11px; margin-left: 8px;">{result_summary}</span>' if result_summary else ''}
                    <span style="color: #666666; font-size: 10px; margin-left: 8px; font-style: italic;">[click to collapse]</span>
                </div>
            </a>
            <div style="
                display: block; 
                margin-top: 4px; 
                padding: 10px; 
                background-color: #2a2a2a; 
                border: 1px solid #555555; 
                border-radius: 4px;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;\n            font-size: 13px;
                font-size: 11px;
                color: #ffffff;
                white-space: pre-wrap;
                max-height: 300px;
                overflow-y: auto;
            ">{escaped_result}</div>
        </div>'''
        else:
            new_tool_html = f'''<div style="display: block; margin: 8px 0; clear: both;" data-tool-id="{tool_result_id}">
            <a href="http://toggle_tool/{tool_result_id}" style="text-decoration: none; color: inherit;">
                <div style="
                    display: block; 
                    padding: 6px 10px; 
                    background-color: rgba(76, 175, 80, 0.1); 
                    border-left: 3px solid {status_color}; 
                    border-radius: 4px; 
                    cursor: pointer;
                    user-select: none;
                " id="{tool_result_id}_header">
                    <span id="{tool_result_id}_arrow" style="color: {status_color}; font-size: 12px; font-weight: 600;">▶</span>
                    <span style="color: {status_color}; font-size: 12px; font-weight: 600; margin-left: 4px;">{status_icon} {tool_name}</span>
                    {f'<span style="color: #888888; font-size: 11px; margin-left: 8px;">{result_summary}</span>' if result_summary else ''}
                    <span style="color: #666666; font-size: 10px; margin-left: 8px; font-style: italic;">[click to expand]</span>
                </div>
            </a>
        </div>'''
        
        # Look for div with data-tool-id attribute
        import re
        
        # Use a more robust pattern that handles nested divs
        def find_matching_div(content, tool_id):
            """Find the outermost div with the specified data-tool-id."""
            start_pattern = rf'<div[^>]*data-tool-id="{re.escape(tool_id)}"[^>]*>'
            
            # Find the start position
            start_match = re.search(start_pattern, content)
            if not start_match:
                return None, None
            
            start_pos = start_match.start()
            end_pos = start_match.end()
            
            # Count nested divs to find the matching closing tag
            div_count = 1
            pos = end_pos
            
            while pos < len(content) and div_count > 0:
                next_open = content.find('<div', pos)
                next_close = content.find('</div>', pos)
                
                if next_close == -1:
                    break
                    
                if next_open != -1 and next_open < next_close:
                    div_count += 1
                    pos = next_open + 4
                else:
                    div_count -= 1
                    if div_count == 0:
                        end_pos = next_close + 6
                        break
                    pos = next_close + 6
            
            if div_count == 0:
                return start_pos, end_pos
            return None, None
        
        # Find the div to replace
        start_pos, end_pos = find_matching_div(html_content, tool_result_id)
        
        if start_pos is not None and end_pos is not None:
            # Replace the found div
            updated_html = html_content[:start_pos] + new_tool_html + html_content[end_pos:]
        else:
            # Simple fallback - look for any div containing the tool_result_id in its attributes
            pattern = rf'<div[^>]*{re.escape(tool_result_id)}[^>]*>.*?</div>'
            updated_html = re.sub(pattern, new_tool_html, html_content, flags=re.DOTALL)
            if updated_html == html_content:
                # Just return the original content if nothing works
                updated_html = html_content
        
        return updated_html
    
    def clear_chat(self):
        """Clear the chat display."""
        # Remove all widgets from the chat layout except the stretch
        while self.chat_layout.count() > 1:
            child = self.chat_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

class PatientDatabaseChatGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.agent = None
        self.agent_worker = None
        self.agent_thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.init_ui()
        self.init_agent()
        self.show_welcome_message()
        
    def init_ui(self):
        """Initialize the user interface with modern design."""
        self.setWindowTitle("Patient Database Chat Assistant")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(1000, 600)
        
        # Create central widget with dark theme
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
            }
        """)
        self.setCentralWidget(central_widget)
        
        # Main layout with better spacing
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 20, 30, 20)
        main_layout.setSpacing(20)
        central_widget.setLayout(main_layout)
        
        # Modern title with gradient background
        title_label = QLabel("Patient Database Chat Assistant")
        # Use system default font for title
        title_font = QApplication.font()
        title_font.setPointSize(24)
        title_font.setWeight(QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setMinimumHeight(80)
        title_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 16px;
                padding: 20px;
                font-family: Arial;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
        """)
        
        # Create horizontal splitter for chat and image viewer
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3d3d3d;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #0078d4;
            }
        """)
        
        # Chat widget
        self.chat_widget = ChatWidget()
        
        # NIfTI viewer widget  
        self.nifti_viewer = NiftiViewer()
        self.nifti_viewer.setMinimumWidth(450)
        # Hide initially when no image is loaded
        self.nifti_viewer.setVisible(False)
        
        # Add widgets to splitter
        content_splitter.addWidget(self.chat_widget)
        content_splitter.addWidget(self.nifti_viewer)
        
        # Set initial sizes (chat takes more space)
        content_splitter.setSizes([700, 450])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Modern progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #0078d4, stop:1 #00bcf2);
                border-radius: 3px;
            }
        """)
        
        # Add widgets to layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(content_splitter, 1)
        main_layout.addWidget(self.progress_bar)
        
        # Connect signals
        self.chat_widget.send_button.clicked.connect(self.send_message)
        self.chat_widget.input_field.returnPressed.connect(self.send_message)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Set modern window properties
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border-top: 1px solid #3d3d3d;
                padding: 5px;
                font-family: Arial;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border-bottom: 1px solid #3d3d3d;
                padding: 4px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #0078d4;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
            }
        """)
        
    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        clear_action = QAction('Clear Chat', self)
        clear_action.setShortcut('Ctrl+L')
        clear_action.triggered.connect(self.clear_chat)
        file_menu.addAction(clear_action)
        
        clear_image_action = QAction('Clear Image', self)
        clear_image_action.setShortcut('Ctrl+I')
        clear_image_action.triggered.connect(self.clear_image_viewer)
        file_menu.addAction(clear_image_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        tool_details_action = QAction('Tool Details', self)
        tool_details_action.setShortcut('Ctrl+T')
        tool_details_action.triggered.connect(self.show_tool_details)
        view_menu.addAction(tool_details_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        examples_action = QAction('Example Queries', self)
        examples_action.triggered.connect(self.show_examples)
        help_menu.addAction(examples_action)
        
    def init_agent(self):
        """Initialize the Agno agent with all tools."""
        try:
            # Initialize the Agno agent with the same configuration as console app
            self.agent = Agent(
                name="Patient Database Agent",
                model=Gemini(id="gemini-2.0-flash", api_key=GOOGLE_API_KEY),
                tools=[
                    # Database tools
                    read_schema,
                    query_database,
                    get_current_datetime,
                    # Patient data tools
                    insert_patient,
                    insert_medical_record,
                    insert_treatment_plan,
                    insert_schedule,
                    update_patient,
                    # Image tools
                    load_patient_image,
                    load_segmentation_mask,
                    # Thinking tools
                    ThinkingTools(add_instructions=True),
                    ReasoningTools(add_instructions=True)
                ],
                # Configure agent parameters
                storage=SqliteStorage(table_name="database_agent", db_file=AGENT_STORAGE),
                add_datetime_to_instructions=True,
                add_history_to_messages=True, 
                num_history_responses=10,
                markdown=True,
                instructions=[
                    "You are a helpful assistant for interacting with the patient radiation therapy database.",
                    "Always use the appropriate tools to interact with the database.",
                    "Before using database modification tools, verify that you have all required information.",
                    "Format responses using tables when displaying multiple records.",
                    "If you're unsure about the database schema, use the read_schema tool.",
                    "For complex queries, break them down into simpler steps.",
                    "Always verify data before inserting or updating (e.g., check dates are in YYYY-MM-DD format).",
                    "Use markdown formatting to structure your responses.",
                    "When returning database query results, format them clearly for easy reading.",
                    "Remember previous conversations with the user and refer back to them when relevant.",
                    "If the user mentions something they asked about earlier, use your conversation memory to provide context.",
                    "CRITICAL: When users ask to view, show, display, or load patient images, you MUST use the load_patient_image tool.",
                    "Examples of image requests: 'load image', 'show image', 'display scan', 'view brain image', 'load default image', 'show patient scan'.",
                    "NEVER use database_query for image requests - ALWAYS use load_patient_image tool for any image-related requests.",
                    "The load_patient_image tool accepts patient_identifier (default: 'image') and image_type (default: 'brain').",
                    "CRITICAL: When users ask for segmentation, tumor detection, brain metastases, or tissue analysis, you MUST use the load_segmentation_mask tool.",
                    "SEGMENTATION KEYWORDS: 'segment', 'segmentation', 'mask', 'tumor', 'metastases', 'overlay', 'detection', 'lesion', 'target' - use load_segmentation_mask tool.",
                    "Examples of segmentation requests: 'segment tumor', 'show brain metastases', 'detect tumor', 'segmentation analysis', 'overlay mask', 'brain tumor segmentation', 'load segmentation'.",
                    "The load_segmentation_mask tool accepts mask_identifier (default: 'mask') and mask_type (default: 'tumor').",
                    "TASK SUBMISSION: For requests to create, submit, or generate new segmentation tasks, use submit_segmentation_task tool.",
                    "TASK KEYWORDS: 'submit task', 'create task', 'generate segmentation', 'new task', 'submit for processing', 'request segmentation', 'task creation' - use submit_segmentation_task tool.",
                    "Examples of task requests: 'submit OAR task for patient 123', 'create BM segmentation task', 'generate new segmentation for patient John', 'submit task for organs at risk', 'create brain metastases task'.",
                    "The submit_segmentation_task tool accepts patient_id (required), image_file_path (optional - auto-detects), and segmentation_type ('OAR' or 'BM').",
                    "WORKFLOW: For segmentation requests, first load an image (if not already loaded), then load the segmentation mask for overlay.",
                    "DO NOT use load_patient_image for segmentation requests - segmentation requires the load_segmentation_mask tool.",
                    "The image viewer supports 3D medical images with slice navigation and segmentation overlays - images and masks will be automatically displayed when loaded.",
                    "Available image files: patient_001_brain.nii.gz (64×64×32), patient_001_ct.nii.gz (80×80×40), image.nii.gz (default 256×256×192).",
                    "Available mask files: targetmaskimage.nii.gz (default segmentation, 256×256×192, 26 labels), brain_tissue_mask.nii.gz (tissue segmentation, 64×64×32)."
                ]
            )
            
            # Create worker thread
            self.agent_thread = QThread()
            self.agent_worker = AgentWorker(self.agent)
            self.agent_worker.moveToThread(self.agent_thread)
            
            # Connect signals
            self.agent_worker.response_ready.connect(self.on_agent_response)
            self.agent_worker.error_occurred.connect(self.on_agent_error)
            self.agent_worker.thinking_update.connect(self.on_thinking_update)
            self.agent_worker.tool_call_started.connect(self.on_tool_call_started)
            self.agent_worker.tool_call_finished.connect(self.on_tool_call_finished)
            # Streaming signals
            self.agent_worker.partial_response.connect(self.on_partial_response)
            self.agent_worker.stream_started.connect(self.on_stream_started)
            self.agent_worker.stream_finished.connect(self.on_stream_finished)
            
            # Start the thread
            self.agent_thread.start()
            
            self.status_bar.showMessage("Agent initialized successfully")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error initializing agent: {str(e)}")
            QMessageBox.critical(self, "Initialization Error", 
                                f"Failed to initialize the agent:\n{str(e)}")
    
    def show_welcome_message(self):
        """Display welcome message in the chat."""
        welcome_msg = """Welcome to the Patient Database Chat Assistant!

This application allows you to interact with the patient database using natural language.

**Example Queries:**
• How many patients are in the database?
• Show me all patients with lung cancer
• Add a new patient named John Smith
• Schedule a treatment appointment for patient X on 2025-10-15

**Database Features:**
• Patient Records: Basic information, demographics, insurance
• Medical Records: Diagnoses, allergies, medical history  
• Treatment Plans: Radiation therapy details, fractions, dosages
• Simulations: CT scan details, contrast usage, immobilization
• Schedules: Appointments, treatment sessions, doctor visits

Type your question below and press Enter or click Send to get started!"""
        
        self.chat_widget.add_message("System", welcome_msg)
        
        # Check if database exists
        if not os.path.exists(DB_FILE):
            warning_msg = f"""[WARNING]: Database file '{DB_FILE}' not found!

Please run `python patient_record_generator.py` first to create the database with sample data.
Database operations will fail until the database is created."""
            self.chat_widget.add_message("System", warning_msg)
    
    def send_message(self):
        """Send user message to the agent."""
        user_input = self.chat_widget.input_field.text().strip()
        
        if not user_input:
            return
            
        # Clear input field
        self.chat_widget.input_field.clear()
        
        # Add user message to chat
        self.chat_widget.add_message("User", user_input, is_user=True)
        
        # Check for greetings
        greeting_keywords = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(user_input.lower().strip() == keyword or 
               user_input.lower().strip() == f"{keyword}!" or
               user_input.lower().strip() == f"{keyword}." 
               for keyword in greeting_keywords):
            # Return greeting response
            greeting_response = "Hello! I'm the Patient Database Agent. How can I help you today? You can ask me questions about patients, their treatments, or other database information."
            self.chat_widget.add_message("Assistant", greeting_response)
            return
        
        # Disable input while processing
        self.chat_widget.input_field.setEnabled(False)
        self.chat_widget.send_button.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_bar.showMessage("Processing your request...")
        
        # Process with agent in thread
        self.executor.submit(self.agent_worker.process_query, user_input)
    
    def on_agent_response(self, response: str):
        """Handle agent response."""
        self.chat_widget.add_message("Assistant", response)
        
        # Re-enable input
        self.chat_widget.input_field.setEnabled(True)
        self.chat_widget.send_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Ready")
        
        # Focus input field
        self.chat_widget.input_field.setFocus()
    
    def on_agent_error(self, error: str):
        """Handle agent error."""
        self.chat_widget.add_message("System", f"❌ Error: {error}")
        
        # Re-enable input
        self.chat_widget.input_field.setEnabled(True)
        self.chat_widget.send_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Error occurred")
    
    def on_thinking_update(self, message: str):
        """Handle thinking update from agent."""
        self.chat_widget.add_thinking_message(message)
    
    def on_tool_call_started(self, tool_name: str, tool_args: str):
        """Handle tool call started event."""
        # Store tool call info for when it finishes
        if not hasattr(self, 'pending_tool_calls'):
            self.pending_tool_calls = {}
        
        # Store with timestamp to handle duplicate tool names
        import time
        call_id = f"{tool_name}_{int(time.time()*1000)}"
        self.pending_tool_calls[call_id] = {
            'name': tool_name,
            'args': tool_args,
            'timestamp': time.time()
        }
        
        # Also store by name for simple lookup (latest call wins)
        self.pending_tool_calls[tool_name] = {
            'name': tool_name,
            'args': tool_args,
            'timestamp': time.time()
        }
    
    def on_tool_call_finished(self, tool_name: str, tool_result: str):
        """Handle tool call finished event."""
        # Get the stored tool args and create the complete block
        tool_args = "[args not available]"
        
        if hasattr(self, 'pending_tool_calls'):
            # First try exact name match
            if tool_name in self.pending_tool_calls:
                call_info = self.pending_tool_calls[tool_name]
                if isinstance(call_info, dict) and 'args' in call_info:
                    tool_args = call_info['args']
                    del self.pending_tool_calls[tool_name]
                else:
                    tool_args = str(call_info)
                    del self.pending_tool_calls[tool_name]
        
        # Handle load_patient_image tool specially - load image in viewer
        if tool_name == "load_patient_image":
            print(f"INFO: Detected load_patient_image tool call, processing...")
            self._handle_image_loading(tool_result)
        elif tool_name == "load_segmentation_mask":
            print(f"INFO: Detected load_segmentation_mask tool call, processing...")
            self._handle_mask_loading(tool_result)
        else:
            print(f"INFO: Tool call detected: {tool_name} (not image-related)")
            
            # Check if this is a database_query that might actually be an image request
            if tool_name == "database_query" and self._is_image_request_response(tool_result):
                print("INFO: Detected potential image request via database_query, attempting image load...")
                # Try to load default image
                QTimer.singleShot(0, lambda: self._load_image_on_main_thread("images/image.nii.gz"))
        
        # Check if we're currently streaming and add to stream, otherwise create separate block
        if hasattr(self.chat_widget, 'streaming_message_id') and self.chat_widget.streaming_message_id:
            self.chat_widget.add_tool_call_to_stream(tool_name, tool_args, tool_result, collapsed=True)
        else:
            self.chat_widget.add_tool_call_block(tool_name, tool_args, tool_result)
    
    def _handle_image_loading(self, tool_result: str):
        """Handle image loading from tool result."""
        try:
            import json
            
            # Parse the tool result 
            if isinstance(tool_result, str):
                if tool_result.strip().startswith('{'):
                    result_data = json.loads(tool_result)
                else:
                    # If it's not JSON, it might be a direct message
                    return
            else:
                result_data = tool_result
            
            # Check if the tool execution was successful
            if isinstance(result_data, dict) and result_data.get("status") == "success":
                file_path = result_data.get("file_path")
                
                if file_path and os.path.exists(file_path):
                    # Use QTimer to ensure this runs on the main GUI thread
                    QTimer.singleShot(0, lambda: self._load_image_on_main_thread(file_path))
                    
                    # Update status immediately
                    self.status_bar.showMessage(f"Loading image: {os.path.basename(file_path)}", 2000)
                else:
                    self.status_bar.showMessage("Image file not found", 3000)
            else:
                # Tool failed or returned error
                error_msg = result_data.get("message", "Unknown error") if isinstance(result_data, dict) else str(result_data)
                self.status_bar.showMessage(f"Image loading failed: {error_msg}", 5000)
                
        except Exception as e:
            self.status_bar.showMessage(f"Error processing image result: {str(e)}", 5000)
    
    def _handle_mask_loading(self, tool_result: str):
        """Handle mask loading from tool result."""
        try:
            import json
            
            # Parse the tool result 
            if isinstance(tool_result, str):
                if tool_result.strip().startswith('{'):
                    result_data = json.loads(tool_result)
                else:
                    return
            else:
                result_data = tool_result
            
            # Check if the tool execution was successful
            if isinstance(result_data, dict) and result_data.get("status") == "success":
                file_path = result_data.get("file_path")
                
                if file_path and os.path.exists(file_path):
                    # Use QTimer to ensure this runs on the main GUI thread
                    QTimer.singleShot(0, lambda: self._load_mask_on_main_thread(file_path))
                    
                    # Update status immediately
                    self.status_bar.showMessage(f"Loading mask: {os.path.basename(file_path)}", 2000)
                else:
                    self.status_bar.showMessage("Mask file not found", 3000)
            else:
                # Tool failed or returned error
                error_msg = result_data.get("message", "Unknown error") if isinstance(result_data, dict) else str(result_data)
                self.status_bar.showMessage(f"Mask loading failed: {error_msg}", 5000)
                
        except Exception as e:
            self.status_bar.showMessage(f"Error processing mask result: {str(e)}", 5000)
    
    def _is_image_request_response(self, tool_result: str) -> bool:
        """Check if a tool result suggests this was actually an image request."""
        try:
            # Look for keywords that suggest this was meant to be an image loading operation
            image_keywords = [
                "loaded", "image", "brain", "default", "display", "viewer", "scan", "medical"
            ]
            
            # Convert to lowercase for case-insensitive search
            result_lower = str(tool_result).lower()
            
            # Count how many image-related keywords appear
            keyword_count = sum(1 for keyword in image_keywords if keyword in result_lower)
            
            # If multiple image keywords appear, this might be an image request
            return keyword_count >= 2
            
        except Exception:
            return False
    
    def _load_image_on_main_thread(self, file_path: str):
        """Load image on the main GUI thread."""
        try:
            success = self.nifti_viewer.load_image(file_path)
            
            if success:
                # Update status to indicate image was loaded
                self.status_bar.showMessage(f"Image loaded: {os.path.basename(file_path)}", 3000)
            else:
                self.status_bar.showMessage("Failed to load image in viewer", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"Error loading image: {str(e)}", 3000)
    
    def _load_mask_on_main_thread(self, file_path: str):
        """Load segmentation mask on the main GUI thread."""
        try:
            success, message = self.nifti_viewer.load_mask(file_path)
            
            if success:
                # Update status to indicate mask was loaded
                self.status_bar.showMessage(f"Mask loaded: {os.path.basename(file_path)}", 3000)
            else:
                self.status_bar.showMessage(f"Failed to load mask: {message}", 3000)
        except Exception as e:
            self.status_bar.showMessage(f"Error loading mask: {str(e)}", 3000)
    
    def clear_image_viewer(self):
        """Clear the image viewer and hide it."""
        self.nifti_viewer.clear_image()
        self.status_bar.showMessage("Image viewer cleared", 2000)
    
    def on_stream_started(self):
        """Handle stream start event."""
        # Initialize streaming message placeholder
        self.streaming_message_id = None
        self.chat_widget.start_streaming_response()
    
    def on_partial_response(self, partial_text: str):
        """Handle partial response from streaming."""
        self.chat_widget.update_streaming_response(partial_text)
    
    def on_stream_finished(self):
        """Handle stream completion event."""
        self.chat_widget.finish_streaming_response()
        
        # Re-enable input (same as on_agent_response)
        self.chat_widget.input_field.setEnabled(True)
        self.chat_widget.send_button.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Ready")
        
        # Focus input field
        self.chat_widget.input_field.setFocus()
    
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_widget.clear_chat()
        self.show_welcome_message()
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Patient Database Chat Assistant

A PyQt6 application that provides a graphical interface for interacting with a patient radiation therapy database using natural language.

Built with:
• PyQt6 for the GUI
• Agno framework for AI agent functionality
• DeepSeek LLM for natural language processing
• SQLite for data storage

Version 1.0"""
        
        QMessageBox.about(self, "About", about_text)
    
    def show_examples(self):
        """Show example queries dialog."""
        examples_text = """Example Queries:

**Database Information:**
• How many patients are in the database?
• What's the schema of the patient table?
• Show me all tables in the database

**Patient Queries:**
• Show me all patients with lung cancer
• Find patients who had treatment in the last month
• List all female patients over 50 years old
• Who has the highest radiation dose?

**Data Management:**
• Add a new patient named John Smith
• Update patient X's phone number to 555-123-4567
• Schedule a treatment appointment for patient X on 2025-10-15
• Mark the appointment on 2025-10-20 as completed

**Analysis Questions:**
• What's the average radiation dose for breast cancer patients?
• Which machine type is used most frequently?
• How many treatment sessions does patient Z have left?
• Compare treatment outcomes between IMRT and VMAT"""
        
        QMessageBox.information(self, "Example Queries", examples_text)
    
    def show_tool_details(self):
        """Show tool details dialog."""
        if not self.chat_widget.tool_details:
            QMessageBox.information(self, "Tool Details", "No tool executions to display yet.")
            return
        
        details_text = "Tool Execution Details:\n\n"
        for tool_id, info in self.chat_widget.tool_details.items():
            details_text += f"Tool: {info['name']}\n"
            details_text += f"Arguments: {info['args']}\n"
            details_text += f"Result: {info['result'][:500]}{'...' if len(info['result']) > 500 else ''}\n"
            details_text += "-" * 50 + "\n\n"
        
        # Create a dialog with scrollable text
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Tool Execution Details")
        dialog.setText("Recent tool executions:")
        dialog.setDetailedText(details_text)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
    
    def closeEvent(self, event):
        """Handle application close event."""
        # Clean up threads
        if self.agent_thread and self.agent_thread.isRunning():
            self.agent_thread.quit()
            self.agent_thread.wait()
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        event.accept()

def main():
    """Main function to run the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Patient Database Chat Assistant")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Configure font for better emoji support
    import platform
    if platform.system() == "Darwin":  # macOS
        # Set default font with emoji fallback  
        font = QFont()
        font.setFamily("Arial")  # Fallback to Arial which supports more symbols
        font.setPointSize(12)
        app.setFont(font)
    
    # Create and show main window
    window = PatientDatabaseChatGUI()
    window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main()