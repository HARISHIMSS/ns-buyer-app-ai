import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline,T5Tokenizer, T5ForConditionalGeneration,M2M100ForConditionalGeneration, M2M100Tokenizer
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from contextlib import asynccontextmanager
import os
from tempfile import NamedTemporaryFile
from spellchecker import SpellChecker
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from geopy.distance import geodesic
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import spacy
import re
from spacy.matcher import Matcher
from spacy.tokens import Span
import numpy as np

__all__ = [torch]
__all__ = [AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline,T5Tokenizer, T5ForConditionalGeneration,M2M100ForConditionalGeneration, M2M100Tokenizer]
__all__ = [Optional]
__all__ = [FastAPI, File, Form, UploadFile]
__all__ = [asynccontextmanager]
__all__ = [os]
__all__ = [NamedTemporaryFile]
__all__ = [SpellChecker]
__all__ = [MongoClient]
__all__ = [TfidfVectorizer]
__all__ = [NearestNeighbors]
__all__ = [geodesic]
__all__ = [linear_kernel]
__all__ = [pd]
__all__ = [spacy]
__all__ = [re]
__all__ = [Matcher]
__all__ = [Span]
__all__ = [np]