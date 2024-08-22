# PyTA
PyTA is a modern, user-friendly alternative to TA-Lib for technical analysis leveraging pandas, numpy and scipy for ease of use. Designed to be compatible with Python 3.10 and later, PyTA provides a comprehensive set of financial indicators and tools without the need for third-party build tools or outdated library versions. Ideal for developers and analysts seeking a straightforward and maintainable solution for the technical analysis of financial data.

## Features
- Modern and User-Friendly: A contemporary alternative to TA-Lib designed for ease of use and integration with modern Python environments.
- Compatibility: Supports Python 3.10 and later versions, ensuring compatibility with recent Python releases.
- Comprehensive Financial Indicators: Provides a wide range of financial indicators and tools essential for technical analysis.
- Dependency-Free: Does not require third-party build tools or outdated libraries, simplifying the installation and setup process.
- Integration with Pandas, Numpy, and Scipy: Leverages these popular libraries for robust and efficient data handling and analysis.
- Straightforward and Maintainable: Offers a clean and maintainable codebase, making it easier for developers and analysts to use and contribute.
- Technical Analysis: Designed specifically for the technical analysis of financial data, offering relevant features and tools for this purpose.

## Installation
From your terminal, use pip to install with the following command:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;`pip install git+https://github.com/saviornt/PyTA`

## Example Usage
1. Once you've installed pyta, import it into your project with `import pyta`
2. Load in your DataFrame, ex: `data`
3. Create a new column that equals a called PyTA Indicator, for example: `data['EMA] = pyta.EMA[data]`

## Example Code
def preprocess_data(data):<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data['EMA'] = pyta.EMA(data)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data['RSI'] = pyta.RSI(data)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;data['VWAP'] = pyta.VWAP(data)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;return data

## Documentation
For more detailed documentation, visit the [Wiki](https://github.com/saviornt/PyTA/wiki).

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/saviornt/PyTA/blob/main/LICENSE) file for details.