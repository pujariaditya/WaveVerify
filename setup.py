#!/usr/bin/env python
"""
Setup configuration for WaveVerify package.

This module handles the installation and packaging of WaveVerify, an audio
watermarking library designed for media authentication and deepfake detection.
It manages dependencies, package discovery, and metadata configuration.

Author: Aditya Pujari and Ajita Rattani
License: MIT (see LICENSE file for details)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from setuptools import setup, find_packages


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================
PACKAGE_NAME: str = "waveverify"
VERSION: str = "0.1.0"
AUTHOR: str = "Aditya Pujari and Ajita Rattani"
AUTHOR_EMAIL: str = ""
DESCRIPTION: str = "Audio watermarking for media authentication and combatting deepfakes"
URL: str = "https://github.com/pujariaditya/WaveVerify"
LICENSE: str = "MIT"
PYTHON_REQUIRES: str = ">=3.8"

# File paths
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
README_PATH: Path = PROJECT_ROOT / "README.md"
REQUIREMENTS_PATH: Path = PROJECT_ROOT / "requirements.txt"

# Package configuration
KEYWORDS: List[str] = [
    "audio", "watermarking", "deepfake", "detection", 
    "authentication", "watermark", "embedding"
]

CLASSIFIERS: List[str] = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Multimedia :: Sound/Audio :: Analysis",
    f"License :: OSI Approved :: {LICENSE} License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
]

PROJECT_URLS: Dict[str, str] = {
    "Bug Tracker": f"{URL}/issues",
    "Documentation": URL,
    "Source Code": URL,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def read_file_content(file_path: Path, encoding: str = "utf-8") -> str:
    """
    Read content from a file with proper error handling.
    
    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        
    Returns:
        Content of the file as string, empty string if file doesn't exist
        
    Raises:
        SystemExit: If critical file reading error occurs
    """
    try:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return ""
            
        with open(file_path, encoding=encoding) as file:
            content = file.read()
            logger.info(f"Successfully read {file_path}")
            return content
            
    except PermissionError as e:
        logger.error(f"Permission denied reading {file_path}: {str(e)}", exc_info=True)
        return ""
        
    except UnicodeDecodeError as e:
        logger.error(f"Encoding error reading {file_path}: {str(e)}", exc_info=True)
        return ""
        
    except Exception as e:
        logger.error(f"Unexpected error reading {file_path}: {str(e)}", exc_info=True)
        sys.exit(1)


def get_requirements(requirements_path: Path) -> List[str]:
    """
    Parse requirements from requirements.txt file.
    
    Args:
        requirements_path: Path to requirements.txt file
        
    Returns:
        List of requirement strings, excluding comments and empty lines
        
    Notes:
        - Filters out empty lines and comments (lines starting with #)
        - Strips whitespace from each requirement
        - Returns empty list if file doesn't exist
    """
    requirements: List[str] = []
    
    try:
        content = read_file_content(requirements_path)
        if not content:
            logger.warning("No requirements.txt found or file is empty")
            return requirements
            
        # Parse requirements line by line
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                requirements.append(line)
                logger.debug(f"Added requirement from line {line_num}: {line}")
                
        logger.info(f"Loaded {len(requirements)} requirements")
        return requirements
        
    except Exception as e:
        logger.error(f"Error parsing requirements: {str(e)}", exc_info=True)
        return []


def get_long_description(readme_path: Path) -> str:
    """
    Read and return the long description from README file.
    
    Args:
        readme_path: Path to README.md file
        
    Returns:
        Content of README file or default description if not found
    """
    long_description = read_file_content(readme_path)
    
    if not long_description:
        logger.warning("README.md not found, using default description")
        return DESCRIPTION
        
    return long_description


def find_packages_config() -> List[str]:
    """
    Configure package discovery with proper includes.
    
    Returns:
        List of packages to include in distribution
        
    Notes:
        Includes main package and all submodules explicitly
    """
    # Define packages to include
    include_patterns = [
        PACKAGE_NAME,
        f"{PACKAGE_NAME}.*",
        "model",
        "model.*",
        "modules", 
        "modules.*",
        "utils",
        "utils.*"
    ]
    
    try:
        packages = find_packages(include=include_patterns)
        logger.info(f"Found {len(packages)} packages: {', '.join(packages[:5])}...")
        return packages
        
    except Exception as e:
        logger.error(f"Error finding packages: {str(e)}", exc_info=True)
        # Fallback to minimal package list
        return [PACKAGE_NAME]


def get_package_data() -> Dict[str, List[str]]:
    """
    Define package data files to include.
    
    Returns:
        Dictionary mapping package names to data file patterns
    """
    return {
        PACKAGE_NAME: ["data/*.yml"],
    }


def get_entry_points() -> Dict[str, List[str]]:
    """
    Define console script entry points.
    
    Returns:
        Dictionary of entry point configurations
        
    Notes:
        Currently empty but structured for future CLI commands
    """
    return {
        "console_scripts": [
            # Future CLI commands will be added here
            # Example: "waveverify=waveverify.cli:main"
        ],
    }


# =============================================================================
# MAIN SETUP CONFIGURATION
# =============================================================================
def main() -> None:
    """
    Execute the setup configuration for WaveVerify package.
    
    This function orchestrates the entire setup process, including:
    - Loading requirements and documentation
    - Configuring package discovery
    - Setting up metadata and classifiers
    - Handling any setup errors gracefully
    """
    try:
        logger.info(f"Starting setup for {PACKAGE_NAME} v{VERSION}")
        
        # Load dynamic content
        requirements = get_requirements(REQUIREMENTS_PATH)
        long_description = get_long_description(README_PATH)
        packages = find_packages_config()
        
        # Execute setup
        setup(
            # Basic metadata
            name=PACKAGE_NAME,
            version=VERSION,
            author=AUTHOR,
            author_email=AUTHOR_EMAIL,
            description=DESCRIPTION,
            long_description=long_description,
            long_description_content_type="text/markdown",
            license=LICENSE,
            
            # URLs
            url=URL,
            project_urls=PROJECT_URLS,
            
            # Classification
            classifiers=CLASSIFIERS,
            keywords=" ".join(KEYWORDS),
            
            # Package configuration
            packages=packages,
            include_package_data=True,
            package_data=get_package_data(),
            
            # Dependencies
            install_requires=requirements,
            python_requires=PYTHON_REQUIRES,
            
            # Entry points
            entry_points=get_entry_points(),
        )
        
        logger.info(f"Setup completed successfully for {PACKAGE_NAME} v{VERSION}")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}", exc_info=True)
        sys.exit(1)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()