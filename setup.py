# Authors: Federico Raimondo <f.raimondo@fz-juelich.de>
#          Sami Hamdan <s.hamdan@fz-juelich.de>
# License: AGPL
from pathlib import Path
import setuptools

version = None
with open(Path('julearn') / '_version.py', 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

with open('README.md', 'r') as fh:
    long_description = fh.read()


DOWNLOAD_URL = 'https://github.com/juaml/julearn/'
URL = 'https://juaml.github.io/julearn'

setuptools.setup(
    name='julearn',
    version=version,
    author='Applied Machine Learning',
    author_email='sami.hamdan@fz-juelich.de',
    description='FZJ AML Library ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    zip_safe=False,
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3'],
    project_urls={
        'Documentation': URL,
        'Source': DOWNLOAD_URL,
        'Tracker': f'{DOWNLOAD_URL}issues/',
    },
    install_requires=['numpy>=1.19.1',
                      'pandas>=1.1.2',
                      'scikit-learn>=0.23.2'],
    python_requires='>=3.6',
)
