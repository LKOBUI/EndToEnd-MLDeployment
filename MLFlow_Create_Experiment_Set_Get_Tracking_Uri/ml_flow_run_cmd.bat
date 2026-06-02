@echo off
setlocal

:: Define directories
set ARTIFACTS_DIR=%CD%\myartifacts
set MLRUNS_DIR=%CD%\mlruns

:: Delete myartifacts folder if it exists
if exist "%ARTIFACTS_DIR%" (
    echo ⚠️ Deleting existing myartifacts directory...
    rmdir /S /Q "%ARTIFACTS_DIR%"
)

:: Delete mlruns folder if it exists
if exist "%MLRUNS_DIR%" (
    echo ⚠️ Deleting existing mlruns directory...
    rmdir /S /Q "%MLRUNS_DIR%"
)

:: Look for any Python file in the current directory
for %%f in (*.py) do (
    set PYFILE=%%f
    goto :found
)

:: If no .py file is found
echo ❌ Error: No Python file (.py) found in the current directory.
exit /b 1

:found
echo ✅ Found Python file: %PYFILE%
echo Running MLflow experiments...

:: Run GLM model
python "%PYFILE%" --experiment_name GLM --model glm --threshold 0.5

:: Run Logistic Regression model
python "%PYFILE%" --experiment_name LOGISTIC --model logistic --C_value 1.0 --l1_ratio 0.7

:: Run MLP model
python "%PYFILE%" --experiment_name MLP --model mlp --hidden_layer 10 --activation relu --solver lbfgs --alpha 0.0001

:: Run XGBoost model
python "%PYFILE%" --experiment_name XGBOOST --model xgboost --learning_rate 0.001

echo ✅ All model runs completed successfully.
endlocal
pause
