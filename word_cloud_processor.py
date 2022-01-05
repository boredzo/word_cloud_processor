#!/usr/bin/python3

import sys
import os
import pathlib
import re
import argparse
import fileinput
import csv
import operator
import collections
from typing import List, Dict, Tuple, Set, Optional

suffix_sets = [
	[ 'ate', 'ates', 'ating', 'ated', 'ation' ],
	[ '', 's', 'es', 'ing', 'ed' ],
	[ 'y', 'ily', 'ier', 'iest', 'ilier', 'iliest' ],
	[ '', 'ly', 'er', 'est', 'lier', 'liest' ],
	[ '', "'s", "s'" ],
	#Theoretically, this list could be turned into a list of suffixes to expand into “___ will”, “____ would”, “____ not”, etc. But all of those words should be in the ignore list anyway, so reconstructing them gains nothing; we might as well simply destroy them here.
	[ '', "'ll", "'d", "'t", "'re", "'ve", "'m" ],
]

def words_equal(word_a, word_b):
	"""Fuzzily compares two words. If they are equal, returns one of them.
	If the two words are exactly equal, you'll get one or the other.
	If the two words differ only in case, you'll get one or the other.
	If the two words have different (English) suffixes, you'll get one or the other.
	If the two words appear to be the same root but only one has a suffix, you'll get the one without the suffix.
	Otherwise, returns None.
	"""
	if word_a == word_b: return word_a
	if word_a.lower() == word_b.lower(): return word_a

	# Stemming
	for suffixes in suffix_sets:
		for i, suffix_a in enumerate(suffixes):
			if word_a.endswith(suffix_a):
				for j, suffix_b in enumerate(suffixes):
					if i == j:
						# We did this comparison already.
						continue

					if word_b.endswith(suffix_b):
						stem_a = word_a[:len(word_a) - len(suffix_a)]
						stem_b = word_b[:len(word_b) - len(suffix_b)]
						if (stem_a == stem_b) or (stem_a.lower() == stem_b.lower()):
							if j < i:
								return word_b
							else:
								return word_a

	return None

def word_trim_suffix(word):
	"Given one word, attempt to detect an English suffix and remove it. If no suffix is detected, return the word as-is."
	for suffixes in suffix_sets:
		for suffix in suffixes[1:]:
			if word.endswith(suffix):
				stem = str(word[:len(word) - len(suffix)])
				return stem + suffixes[0]
	return str(word)

class FuzzyString(str):
	def __hash__(self):
		if not hasattr(self, 'cached_stem'):
			self.cached_stem = word_trim_suffix(self).lower()
		return hash(self.cached_stem)
	def __eq__(self, other):
		return words_equal(str(self), str(other))

class TokenGraphNode(object):
	def __init__(self, label: str):
		self.label = label
		self.children = {}
		self.canonical_node = None

	def add_child_node(self, token):
		key = FuzzyString(token.label)
		try:
			existing_child = self.children[key]
		except KeyError:
			self.children[key] = token
		else:
			if token is not self.children[key]:
				assert key not in self.children, 'Trying to add token that is already present: key {}, existing token {}, new token {}'.format(key, self.children[key], token)

	def add_raw_tokens(self, raw_tokens: List[str], canonical_node: Optional[object]=None):
		"Convenience method for adding a descending series (path) of nodes for tokens from a sequence. If canonical_node is None, creates a new canonical node for this path and returns it. If canonical_node is not None, it will be appended to the path as the canonical node for it, and returned. If a complete path to a canonical node already exists within the graph that matches raw_tokens exactly, no addition to the graph will be made, and the existing canonical node will be returned."

		raw_tokens = list(raw_tokens)
		addition_point, remaining_tokens = self.find_longest_path(raw_tokens)
		if not remaining_tokens:
			# Existing path found! If it doesn't already have a canonical node, add ours; then return whichever one it has.
			if not addition_point.canonical_node:
				addition_point.canonical_node = canonical_node
			return addition_point.canonical_node

		if not addition_point:
			addition_point = self

		graph_path_head = graph_path_tail = None
		for raw_token in remaining_tokens:
			token_node = TokenGraphNode(FuzzyString(raw_token))
			if graph_path_head is None:
				graph_path_head = token_node
			if graph_path_tail is not None:
				graph_path_tail.add_child_node(token_node)
			graph_path_tail = token_node

		if graph_path_head:
			addition_point.add_child_node(graph_path_head)
		if graph_path_tail:
			graph_path_tail.canonical_node = canonical_node
			return canonical_node

	def find_longest_path(self, raw_tokens: List[str], _indent=0):
		"Given a list of strings (raw tokens), search the graph starting from the descendants of this node, building the longest path that is a prefix of the raw_tokens. If at least one token (starting from the first) is matched, returns (deepest_matched_node, remaining_tokens). If all tokens are matched, returns (canonical_node, []). If no tokens are matched, returns (None, raw_tokens)."
		tabs = '\t' * _indent
#		print(tabs + 'Search begins for', raw_tokens, 'in', self)

		raw_tokens = list(raw_tokens)
		if not raw_tokens:
#			print(tabs + 'Search failure A: No tokens to search for')
			return (None, raw_tokens)

		first_token = FuzzyString(raw_tokens[0])
		try:
			matched_children = [ self.children[first_token] ]
		except KeyError:
#			print(tabs + 'Search Failure B: No child matches', first_token)
			return (self, raw_tokens)

		remaining_tokens = raw_tokens[1:]
		if remaining_tokens:
			best_match = None
			best_match_unmatched_tokens = remaining_tokens
			for child in matched_children:
				result_node, unmatched_tokens = child.find_longest_path(remaining_tokens, _indent + 1)
#				print(tabs + 'Recursive search returned', result_node, unmatched_tokens)
				if len(unmatched_tokens) <= len(best_match_unmatched_tokens):
					best_match = result_node
					best_match_unmatched_tokens = unmatched_tokens
#			if not best_match:
#				print(tabs + 'Search Failure C: No path found matching remaining tokens', remaining_tokens)
			return (best_match, best_match_unmatched_tokens)
		else:
			for child in matched_children:
				result = child.canonical_node
#				if not result:
#					print(tabs + 'Search Failure D: Matched child had no canonical node:', result)
				if result:
#					print(tabs + 'Search Success D: Found canonical node:', result)
					return (result, remaining_tokens)
#			print(tabs + 'Search Success E: No remaining tokens; returning self', self)
			return (self, remaining_tokens)

#		print(tabs + 'Search Failure Z: Loop exhausted; end of function reached')
		return (None, raw_tokens)

	def find(self, raw_tokens: List[str]):
		"Given a list of strings (raw tokens), search the graph starting from the descendants of this node. If a path can be found that spells out these raw tokens, returns the canonical node at the end of the path. Otherwise, returns None."
		if not raw_tokens: return None

		first_token = FuzzyString(raw_tokens[0])
		try:
			matched_children = [ self.children[first_token] ]
		except KeyError:
			return None

		remaining_tokens = raw_tokens[1:]
		if remaining_tokens:
			for child in matched_children:
				result = child.find(remaining_tokens)
				if result:
					return result
		else:
			for child in matched_children:
				result = child.canonical_node
				if result:
					return result

		return None

	def reprtree(self, _indent=0):
		"Returns a string that represents the entire hierarchy under this node."
		lines = []
		lines.append('\t' * _indent + repr(self))
		_indent += 1
		for key, child in self.children.items():
			lines.append('\t' * _indent + key)
			lines.extend(child.reprtree(_indent))
		_indent -= 1
		if _indent > 0:
			return lines
		else:
			return '\n'.join(lines)

	def __repr__(self):
		if self.canonical_node:
			return '<{} {} canonical={}>'.format('TokenGraphNode', repr(self.label), repr(self.canonical_node.label))
		else:
			return '<{} {} with {} children {}>'.format('TokenGraphNode', repr(self.label), len(self.children), repr(list(self.children)))

	def __getitem__(self, key):
		"Given a raw token, return the child node matching that raw token, or None if there is no such child."
		try:
			return self.children[key]
		except KeyError:
			return None

def raw_tokens_from_string(s: str,
	_exp=re.compile('\\w+(\'[a-z]+)?'), #Note: '[letters] pulls in 's, 'll, 'd, etc.
	_hashtagExp=re.compile('[#@](\\w+)'), #Also handles Twitter handles (e.g., @TheDemocrats)
	_hashtagWordExp=re.compile('[A-Z][a-z]+|[A-Z]*[0-9]+'),
	_anythingInterestingExp=re.compile('[#@\\w]')
):
	"Generates a sequence of raw tokens suitable for looking up in a token graph, or building token graph nodes from."
	s = s.strip()
	while s:
		match = _exp.match(s)
		if match:
			s = s[match.end() + 1:]
			word = match.group(0)
			apostrophe_index = word.find("'")
			if apostrophe_index >= 0:
				word = word[:apostrophe_index]
			yield word
		else:
			match = _hashtagExp.match(s)
			if match:
				tag = match.group(1)
				for match in _hashtagWordExp.finditer(tag):
					yield match.group(0)
				s = s[match.end() + 1:]
			else:
				# TODO: Strip punctuation
				match = _anythingInterestingExp.search(s)
				if match:
					s = s[match.start():]
				else:
					#Nothing but garbage left. Bail.
					break

class TokenRecognizer(object):
	"""An object that can be fed one raw token at a time. As tokens are consumed, the recognizer crawls the token graph from the root supplied at initialization, attempting to find canonical nodes matching sequences of tokens and call its callback with the canonicalized tokens, and calling the callback with any tokens that could not be canonicalized.
	As tokens are consumed, the recognizer descends with each token matched by the current path so far. When a token is not found among the current node's descendants, the current node is checked for a canonical node; if such is found, the callback is called with that node's label; if not, the callback is called once for each token in the path, and the path reset. When no path is current and a token is not found in the root of the graph, the callback is immediately called with that token.
	(This is, essentially, TokenGraphNode.find but broken up.)
	When the .end method is called, this, too, flushes the path and calls the callback with each token in the path.
	The callback should be a function that can be called with one argument, which is a string.
	"""
	def __init__(self, graph_root: TokenGraphNode, callback: callable):
		self.graph_root = graph_root
		self.callback = callback
		#The path
		self.raw_tokens_recognized = []
		self.current_graph_node = None

	def flush(self):
		"Walks the accumulated list of raw tokens and calls the callback for each one, then empties the list."
		for raw_token in self.raw_tokens_recognized:
			self.callback(raw_token)
		del self.raw_tokens_recognized[:]

	def consume(self, raw_token: str):
		if self.current_graph_node:
			child_node = self.current_graph_node[raw_token]
			if not child_node:
				if self.current_graph_node.canonical_node:
					self.callback(self.current_graph_node.canonical_node)
				else:
					self.flush()
				self.raw_tokens_recognized[:] = [ raw_token ]
				self.current_graph_node = self.graph_root[raw_token]
		else:
			self.current_graph_node = self.graph_root[raw_token]
			if not self.current_graph_node:
				self.callback(raw_token)
			else:
				self.raw_tokens_recognized[:] = [ raw_token ]
		
	def end(self):
		if self.current_graph_node.canonical_node:
			self.callback(self.current_graph_node.canonical_node)
		else:
			self.flush()
		self.current_graph_node = None

class TokenCollector(object):
	"Used for testing the TokenRecognizer. Exposes a method that can be used as a recognizer's callback, and accumulates a list of tokens (self.raw_tokens) that can then be compared against the input or an expected variation on it."
	def __init__(self):
		self.raw_tokens = []

	def collect(self, raw_token: str):
		self.raw_tokens.append(raw_token)

class WordHistogramAccumulator(object):
	def __init__(self):
		self.counts = collections.Counter()
		self.ignore_list = set()

	def add_word(self, word: str):
		word = FuzzyString(word)
		if word not in self.ignore_list:
			self.counts[word] += 1

	def ignore(self, words: Set[FuzzyString]):
		"Given a set of words, add them to the set of words to be ignored."
		self.ignore_list.update(words)

	def __iter__(self):
		return iter(self.counts.items())

def main(opts):
	synonyms_graph = TokenGraphNode('<root>')
	if opts.synonyms_csv_path:
		syn_csv = csv.reader(open(opts.synonyms_csv_path, 'r'))
		next(syn_csv)
		for row in syn_csv:
			# Each item on the row is either a word or phrase or hashtag that should be considered one of the elements in this synonym group, or the empty string.
			# Each non-empty item must be added to the synonym graph, like so:
			"""
			Bipartisan -> Infrastructure -> Law --------> .
			                   \                          ^
			                    \                         |
			                     \--------> Framework ----|
			                      \-------> Bill ---------|
			                       \------> Act ----------|
			Biden -> Infrastructure -> Law  --------------|
			                     \--------> Framework ----|
			                      \-------> Bill ---------|
			                       \------> Act ----------
			"""
			# Note that all of the synonyms arrive at the same end-of-phrase token. This token is labeled with the full canonical term (taken from the first entry in the row) and is used when exporting the histogram.
			first_variant = row[0]
			final_token = TokenGraphNode(first_variant)
			for variant in row:
				raw_tokens = list(raw_tokens_from_string(variant))
				if not raw_tokens:
					# Not every row will fill out all columns (because the rows don't all have equal numbers of synonyms). Ignore empty cells.
					continue
				synonyms_graph.add_raw_tokens(raw_tokens, final_token)

	ignore_list = set()
	if opts.ignore_list_path:
		ignore_file = open(opts.ignore_list_path, 'r')
		for line in ignore_file:
			idx = line.find('#')
			if idx >= 0:
				line = line[idx:]
			line = line.strip()
			for raw_token in raw_tokens_from_string(line):
				ignore_list.add(FuzzyString(raw_token))

	texts_csv_file = open(opts.input_csv_path, 'r') if opts.input_csv_path else sys.stdin
	texts_csv = csv.reader(texts_csv_file)
	header = next(texts_csv)
	try:
		text_idx = header.index('Source text')
	except ValueError:
		text_idx = 0

	histogram = WordHistogramAccumulator()
	histogram.ignore(ignore_list)
	recognizer = TokenRecognizer(synonyms_graph, histogram.add_word)

	for row in texts_csv:
		text = row[text_idx]
		for i, raw_token in enumerate(raw_tokens_from_string(text)):
			recognizer.consume(raw_token)

	output_csv = csv.writer(sys.stdout)
	output_csv.writerow([ 'Count', 'Word' ])
	for word, count in histogram:
		output_csv.writerow([ count, word ])

def self_test():
	no_input_value = object()
	class TestCase(object):
		def __init__(self, description: str, expected, test_func=None, input_value=no_input_value):
			self.description = description
			self.expected = expected
			self.input_value = input_value if input_value is not no_input_value else expected
			self.test_func = test_func

			self.checker = operator.eq
			self.ran = False
			self.passed = None

		def check(self):
			if callable(self.test_func):
				try:
					self.check_result(self.test_func(self))
				except:
					self.passed = False
					self.ran = True
					raise

		def check_result(self, actual):
			self.actual = actual
			self.passed = self.checker(self.expected, self.actual)
			self.ran = True

		def __str__(self):
			if self.passed:
				result_emoji = '\u2705' # White check mark on green
			elif self.passed is False:
				result_emoji = '\u274c' # Red X
			else:
				result_emoji = '\u2753' # Red question mark

			return '{} {}: Expected {}, got {}'.format(result_emoji, self.description, repr(self.expected), repr(self.actual))

	tests_passed = 0
	tests_ran = 0

	graph = TokenGraphNode('<root>')
	canonical_node = TokenGraphNode('#BidenInfrastructureLaw')
	graph.add_raw_tokens([ 'Biden', 'Infrastructure', 'Law' ], canonical_node)
	graph.add_raw_tokens([ 'Bipartisan', 'Infrastructure', 'Law' ], canonical_node)
	graph.add_raw_tokens([ 'Biden', 'Infrastructure', 'Act' ], canonical_node)

	def test_token_accumulator(test):
		#Sadly, this one can't be a lambda, so it's up here.
		acc = TokenCollector()
		rec = TokenRecognizer(graph, acc.collect)
		for raw_token in test.expected:
			rec.consume(raw_token)
		return acc.raw_tokens

	tests = [
		#raw_tokens_from_string
		TestCase('Raw tokens from empty string', [ ], lambda test: list(raw_tokens_from_string(''))),
		TestCase('Raw tokens from hashtag', [ 'Black', 'Lives', 'Matter' ], lambda test: list(raw_tokens_from_string('#BlackLivesMatter'))),
		TestCase('Raw tokens from hashtag + Twitter handle', [ 'CA27', 'officially', 'rated', 'a', 'Toss', 'Up', 'by', 'Cook', 'Political' ], lambda test: list(raw_tokens_from_string('''#CA27 officially rated a Toss Up by 
@CookPolitical'''))),
		TestCase('Raw tokens from sentence including possessive', [ 'I', 'love', 'Kyle', 'barbecue' ], lambda test: list(raw_tokens_from_string("I love Kyle's barbecue"))),
		TestCase('Raw tokens from simple string', [ 'Black', 'lives', 'matter' ], lambda test: list(raw_tokens_from_string('Black lives matter'))),
		TestCase('Raw tokens from hyphenated string', [ 'Black', 'lives', 'matter' ], lambda test: list(raw_tokens_from_string('Black-lives-matter'))),
		TestCase('Raw tokens from punctuation-riddled string', [ 'Black', 'lives', 'matter' ], lambda test: list(raw_tokens_from_string('"Black! lives! matter!!!"'))),
		#words_equal
		TestCase('Equality of equal words', 'blah', lambda test: words_equal('blah', 'blah')),
		TestCase('Equality of case-insensitively equal words', True, lambda test: bool(words_equal('blah', 'Blah'))),
		TestCase('Inequality of unrelated words', None, lambda test: words_equal('blah', 'halb')),
		TestCase('Equality of noun and its plural', 'tool', lambda test: words_equal('tool', 'tools')),
		TestCase('Equality of verb and its gerund', 'tool', lambda test: words_equal('tooling', 'tool')),
		TestCase('Equality of two different suffixes on the same root', True, lambda test: bool(words_equal('tools', 'tooling'))),
		TestCase('Inequality of unrelated words with same suffix', None, lambda test: words_equal('walking', 'talking')),
		#TokenGraphNode
		TestCase('Searching a token graph for the canonical phrase', canonical_node, lambda test: graph.find([ 'Biden', 'Infrastructure', 'Law' ])),
		TestCase('Searching a token graph for the canonical phrase in different case', canonical_node, lambda test: graph.find([ 'biden', 'infrastructure', 'law' ])),
		TestCase('Searching a token graph for a non-canonical phrase from the start', canonical_node, lambda test: graph.find([ 'Bipartisan', 'Infrastructure', 'Law' ])),
		TestCase('Searching a token graph for a non-canonical phrase from the end', canonical_node, lambda test: graph.find([ 'Biden', 'Infrastructure', 'Act' ])),
		TestCase('Searching a token graph for a phrase that is incompletely present', None, lambda test: graph.find([ 'Biden', 'Infrastructure', 'Piano' ])),
		TestCase('Searching a token graph for a phrase that is not present at all', None, lambda test: graph.find([ 'Patient', 'Protection', 'and', 'Affordable', 'Care', 'Act' ])),
		#TokenRecognizer
		TestCase('Recognizing word tokens', [ 'Voting', 'rights', 'now' ], test_token_accumulator),
		TestCase('Recognizing a hashtag token in a phrase', [ '#BidenInfrastructureLaw' ], test_token_accumulator, input_value=[ 'Biden', 'Infrastructure', 'Act' ]),
		TestCase('Recognizing tokens in a sentence', [ 'Congratulations', 'to', 'Senate', 'Democrats', 'on', 'passing', 'the', '#BidenInfrastructureLaw' ], test_token_accumulator, input_value=[ 'Congratulations', 'to', 'Senate', 'Democrats', 'on', 'passing', 'the', 'Bipartisan', 'Infrastructure', 'Law' ]),
		TestCase('Recognizing word tokens from a partial match', [ 'Celebrating', 'the', 'passage', 'of', 'the', 'Biden', 'Infrastructure', 'Deed' ], test_token_accumulator),
TestCase('Recognizing word tokens from a barely-started match', [ 'Congratulating', 'the', 'Biden', 'Administration', 'on', 'the', 'economic', 'recovery' ], test_token_accumulator),
	]
	try:
		for t in tests:
			t.check()

			tests_passed += 1 if t.passed else 0
			tests_ran += 1 if t.ran else 0
	finally:
		for t in tests:
			print(t)
		else:
			print('{} of {} tests PASSED'.format(tests_passed, tests_ran))
			tests_failed = tests_ran - tests_passed
			if tests_failed != 0:
				print('\u203c {} of {} tests FAILED'.format(tests_failed, tests_ran))

if __name__ == '__main__':	
	parser = argparse.ArgumentParser()
	parser.add_argument('--synonyms', dest='synonyms_csv_path', default=None, type=pathlib.Path, help='Path to a CSV file listing synonyms. Each row is one group of synonyms; each item in the row is one term that is synonymous with all other terms in the group.')
	parser.add_argument('--ignore', dest='ignore_list_path', default=None, type=pathlib.Path, help='Path to a text file listing words to ignore. Each line is one singular word; no phrases are allowed. This should generally be a list of words like "the" and "an" that you don\'t care about. You can usually generate this list by selecting the most frequent words in a sufficiently large corpus.')
	parser.add_argument('--self-test', default=False, action='store_true', help='Run internal self-tests. For development only.')
	parser.add_argument('input_csv_path', metavar='input.csv', default=None, type=pathlib.Path, help='Path to a CSV file containing source texts. The "Source text" column, if a column is so named, will be used; otherwise, the first column will be used.')
	opts = parser.parse_args()

	if opts.self_test:
		self_test()
	else:
		main(opts)	
