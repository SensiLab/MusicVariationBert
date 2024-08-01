from setuptools import setup

setup(
    name='MusicVariationBert',
    version='0.1.0',    
    description='MusicBert based music variation',
    url='https://github.com/SensiLab/MusicVariationBert',
    author='Stephen Krol',
    author_email='stephen.krol@monash.edu',
    license='BSD 2-clause',
    packages=['MusicVariationBert'],
    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
    ],
)