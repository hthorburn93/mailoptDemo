# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:59:06 2021

@author: thorburh
"""


from setuptools import setup, find_packages
#pkg_resources


setup(name='mailopt',
      version='0.1.0',
      author='Hamish Thorburn',
      author_email='h.thorburn@lancaster.ac.uk',
      url='https://github.com/hthorburn93/Base-T-Expanded-Network',
      description="Package for optimizing staffing levels at mail centres",
      packages=find_packages(exclude=('tests', 'scripts')),
      include_package_data=True,
      package_data={'mailopt': ['Data/*.csv']}
)
