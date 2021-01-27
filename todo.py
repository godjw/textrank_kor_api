#-*- coding:utf-8 -*-
from flask import request, jsonify
from flask_restx import Resource, Api, Namespace

from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
from collections import Counter


class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.okt = Okt()
        # 불용어
        self.stopwords = ()

    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

    # 텍스트에서 명사 추출
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join(
                    [noun for noun in self.okt.nouns(str(sentence)) if noun not in self.stopwords and len(noun) > 1]))
        return nouns

    # 텍스트에서 숫자 추출
    def get_numbers(self, sentences):
        numbers = []
        for sentence in sentences:
            temp = sentence.split()
            for num, pos in self.okt.pos(str(sentence)):
                if pos == 'Number':
                    for word in temp:
                        if num in word:
                            numbers.append(word)
        return numbers

    # 명사의 빈도수 저장
    def get_noun_count(self, text):
        nouns = self.okt.nouns(text)
        for i, v in enumerate(nouns):
            if len(v) < 2:
                nouns.pop(i)
        count = Counter(nouns)
        noun_count = count.most_common(10)
        return noun_count


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence

    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85):  # d = damping factor
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0  # diagonal 부분을 0으로
            link_sum = np.sum(A[:, id])  # A[:, id] = A[:][id]
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1

        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = b
        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        self.numbers = self.sent_tokenize.get_numbers(self.sentences)
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)
        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
        self.noun_count = self.sent_tokenize.get_noun_count(text)

    def summarize(self, sent_num):
        summary = []
        index = []
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        return summary

    def keywords(self, word_num=12):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)
        keywords = []
        index = []
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)
        # index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])
        return keywords


# 키워드 추출할 텍스트 입력
# url = "https://ko.wikipedia.org/wiki/%EC%9C%84%ED%2%A4%EB%B0%B1%EA%B3%BC"

Todo = Namespace('Todo')
@Todo.route('')
class TodoPost(Resource):
    def post(self):
        global text_str
        text_str = request.json.get('data')
        textrank = TextRank(text_str)
        n = 5
        i = 1
        summary = []
        for row in textrank.summarize(n):
            summary.append(row)
            i += 1
        keywords = textrank.keywords()
        noun_list = textrank.nouns
        noun_set = set()

        for n in noun_list:
            temp_list = n.split()
            for word in temp_list:
                noun_set.add(word)
        noun_set = list(noun_set)
        numbers = textrank.numbers

        question_list = {}
        ans_candidate = list(set(keywords + numbers + noun_set[:20]))

        for sentence in summary:
            sentence = sentence.split()
            question = []
            answer = []
            i = 1
            for word in sentence:
                if word in ans_candidate and i < 5:
                    if word in answer:
                        question.append("[   " + str(answer.index(word) + 1) + "   ]")
                    else:
                        answer.append(word)
                        question.append("[   " + str(i) + "   ]")
                    i = i + 1
                else:
                    question.append(word)
            question_list[(' '.join(question))] = answer
        return {
            "data" : question_list
        }
