# mT5 Summarization App
Streamlit app for summarizing news articles with mT5 &amp; XLSum.

![Screenshot of App](img/screenshot_main.PNG)

## To Create The Environment
```bash
conda env create -n ENVNAME -f environment.yml
```

## To Run The App
```bash
conda activate ENVNAME
streamlit run app.py
```
Click one of these links:
```bash
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://XXX.XXX.X.XXX:8501
```

## Acknowledgements & Links
* Model trained by [BUET CSE NLP Group](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum).
* [mT5 Model](https://github.com/google-research/multilingual-t5)
* [XL-Sum Dataset](https://aclanthology.org/2021.findings-acl.413/); dataset has a [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
