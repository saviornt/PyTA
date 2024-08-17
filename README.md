# PyTA
PyTA is a modern, user-friendly alternative to TA-Lib for technical analysis leveraging pandas, numpy and scipy for ease of use. Designed to be compatible with Python 3.10 and later, PyTA provides a comprehensive set of financial indicators and tools without the need for third-party build tools or outdated library versions. Ideal for developers and analysts seeking a straightforward and maintainable solution for technical analysis in financial data.

# Installation
From your terminal, use pip to install with the following command:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;`pip install git+https://github.com/saviornt/PyTA`

# Example Usage
import pyta<br/>
def preprocess_data(data):<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data['EMA'] = pyta.EMA(data)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data['RSI'] = pyta.RSI(data)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data['VWAP'] = pyta.VWAP(data)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;return data
