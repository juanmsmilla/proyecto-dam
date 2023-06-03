

HIDE_ST_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stFileUploadDropzone"] div div::before {content:"Arrastre aquí su documento PDF"}
    [data-testid="stFileUploadDropzone"] div div span{display:none;}
    [data-testid="stFileUploadDropzone"] div div::after {font-size: .8em; content:"Tamaño límite - 20MB"} 
    </style>
    """