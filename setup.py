from setuptools import setup, find_packages

setup(
    name='clue-python',
    version='0.0.1',
    packages=find_packages(),
    description='Python implementation of CLUE (https://arxiv.org/abs/2409.03021)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Nikil Patel',
    author_email='nikil.patel@example.com',
    url='https://github.com/nikilpatel94/CLUE-python',
    install_requires=[
        "python-dotenv==1.0.1",
        "transformers==4.47.1",
        "pandas==2.2.3",
        "torch==2.5.1",
        "openai==1.58.1",
        "groq==0.13.1",
        "datasets==3.2.0",
        "numpy==2.2.4",
        "sentence-transformers==4.0.2",
        "pydantic==2.10.6",
        "instructor==1.7.9",
        "appdirs==1.4.4",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
