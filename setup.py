from setuptools import find_packages, setup

setup(

    name = "Credit Risk Modeling",
    version = "0.0.1",
    author = "Gaurav Jaiswal",
    author_email= "jaiswalgaurav863@gmail.com",
    install_requirements = ["scikit-learn", "scipy", "statsmodels"],
    packages=find_packages()

)