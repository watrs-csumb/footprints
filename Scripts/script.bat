@ECHO OFF
ECHO proceeding to install dependencies...

SET PYTHON_PATH=python

REM Create virtual environment
IF NOT EXIST venv (
    ECHO Creating virtual environment...
    %PYTHON_PATH% -m venv venv
) ELSE (
    ECHO Virtual environment already exists.
)

REM Activate virtual environment
ECHO Activating virtual environment...
CALL venv\Scripts\activate.bat
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment.
    EXIT /B 1
)

ECHO Virtual environment activated.
ECHO Installing dependencies...
CALL pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO Failed to install dependencies.
    EXIT /B 1
)

ECHO Done!
