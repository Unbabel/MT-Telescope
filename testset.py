import streamlit as st
from io import StringIO
import pandas as pd
import stanza
        

STANZA_NER_LANGS = [
    "ar",
    "zh",
    "nl",
    "en",
    "fr",
    "de",
    "ru",
    "uk"    
]

@st.cache()
def read_lines(file):
    if file is not None:
        file = StringIO(file.getvalue().decode())
        lines = [line.strip() for line in file.readlines()]
        return lines
    return None

class PairedTestset:

    def __init__(
        self, 
        sources: list, 
        system_x: list, 
        system_y: list, 
        references: list, 
        src_lang: str = None, 
        trg_lang: str = None
    ) -> None:
        self.sources = sources
        self.references = references
        self.system_x = system_x
        self.system_y = system_y
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        assert len(references) == len(sources), "mismatch between references and sources ({} > {})".format(len(references), len(sources))
        assert len(system_x) == len(system_y), "mismatch between system x and system y ({} > {})".format(len(system_x), len(system_y))
        assert len(system_x) == len(references), "mismatch between system x and references ({} > {})".format(len(system_x), len(references))

    @classmethod
    def read_data(cls):
        st.sidebar.subheader("Upload Files:")
        source_file = st.sidebar.file_uploader("Upload Sources", type=["txt"])
        sources = read_lines(source_file)

        ref_file = st.sidebar.file_uploader("Upload References", type=["txt"])
        references = read_lines(ref_file)
        
        x_file = st.sidebar.file_uploader("Upload System X Translations", type=["txt"])
        x = read_lines(x_file)

        y_file = st.sidebar.file_uploader("Upload System Y Translations", type=["txt"])
        y = read_lines(y_file)

        if (
            (ref_file is not None)
            and (source_file is not None)
            and (y_file is not None)
            and (x_file is not None)
        ):

            st.success("Source, Refereces and Translations were successfully uploaded!")
            return cls(
                sources, 
                x, y, 
                references, 
                source_file.name.split(".")[-2], 
                ref_file.name.split(".")[-2]
            )
        else:
            st.info(
                "Please use the sidebar to upload Source, Reference and the Translations"  
                " you wish to compare."
            )

    def __len__(self):
         return len(self.references)

    def __getitem__(self, i):
        return self.sources[i], self.system_x[i], self.system_y[i], self.references[i]

    @st.cache()
    def length_filter(self, min_value, max_value):
        with st.spinner("Applying length filter..."):
            dataframe = pd.DataFrame()
            dataframe["ref"] = self.references
            dataframe["lengths"] = [len(ref) for ref in list(dataframe["ref"])]
            dataframe["bins"], retbins = pd.qcut(
                dataframe["lengths"], len(range(0,100,5)), labels=range(0,100,5), retbins=True
            )
            length_buckets = list(dataframe["bins"])

            src = [
                self.sources[i] 
                for i in range(len(self)) 
                if (length_buckets[i] >= min_value and length_buckets[i] <= max_value)
            ]            
            ref = [
                self.references[i]
                for i in range(len(self)) 
                if (length_buckets[i] >= min_value and length_buckets[i] <= max_value)    
            ]            
            x = [
                self.system_x[i] 
                for i in range(len(self)) 
                if (length_buckets[i] >= min_value and length_buckets[i] <= max_value)    
            ]
            y = [
                self.system_y[i] 
                for i in range(len(self)) 
                if (length_buckets[i] >= min_value and length_buckets[i] <= max_value)    
            ]
            return PairedTestset(
                sources=src,
                system_x=x,
                system_y=y,
                references=ref,
                src_lang=self.src_lang,
                trg_lang=self.trg_lang
            )

    def check_stanza_languages(self):
        if self.src_lang in STANZA_NER_LANGS:
            return True
        elif self.trg_lang in STANZA_NER_LANGS:
            return True
        else:
            st.warning(f"source and target languages not supported by stanza NER library.")
            return False

    @st.cache()
    def ner_filter(self):
        with st.spinner("Applying Named Entities filter..."):
            if not self.src_lang in STANZA_NER_LANGS:
                language = self.trg_lang
                segments = self.references
            else:
                language = self.src_lang
                segments = self.sources

            stanza.download(language)
            nlp = stanza.Pipeline(lang=language, processors='tokenize,ner')

            segments_with_ne = []
            for i, segment in enumerate(segments):
                doc = nlp(segment)
                if doc.ents:
                    segments_with_ne.append(i)

            return PairedTestset(
                sources=[self.sources[i] for i in segments_with_ne],
                system_x=[self.system_x[i] for i in segments_with_ne],
                system_y=[self.system_y[i] for i in segments_with_ne],
                references=[self.references[i] for i in segments_with_ne],
                src_lang=self.src_lang,
                trg_lang=self.trg_lang
            )
        
