from setuptools import find_packages, setup

setup(name='jaw',
      version='0.1',
      url='https://github.com/Yvregon/JAW',
      license='GNU General Public License v3.0',
      author='Quentin Duupré',
      author_email='quentin.mathias.dupre@gmail.com',
      description='Just Another (Pytorch) Wrapper',
      packages=find_packages(),
      long_description=open('README.md').read(),
      zip_safe=False,
    )
