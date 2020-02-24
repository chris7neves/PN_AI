1) Add your python version to Path (e.g. C:\Users\Massimo\AppData\Local\Programs\Python\Python37) and add Scripts to path too (e.g. C:\Users\Massimo\AppData\Local\Programs\Python\Python37\Scripts)
2) Open up a cmd.exe and navigate to the folder in your project containing the PipFile and PipFile.lock
3) Type "pipenv install --python python" (if you haven't added python to your path you can also do "pipenv install --python *path_to_python_version_executable*")
This will create a virtual environment that contains the dependencies specified within the PipFiles. If packages are added, the PipFile is automatically updated
4) Launch the virtual env by typing "pipenv shell"
5) type "pip freeze" to confirm that the packages match up with the ones defined in your PipFile. 

ADDING A PACKAGE
6) type "pipenv install *package_name*" (e.g. "pipenv install django")
