cimport cython
from cpython.bytes cimport PyBytes_FromStringAndSize


cdef extern from "Python.h":
    char* PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t *size)


@cython.boundscheck(False)
@cython.wraparound(False)
def n_grams(list tokens, int n):
    cdef int n_tokens = len(tokens)
    cdef set result = set()
    cdef int i
    cdef bytes window
    cdef list window_tokens

    for i in range(n_tokens - n):
        window_tokens = tokens[i:i + n]
        window = b''.join(window_tokens)
        result.add(window)
    return list(result)

@cython.boundscheck(False)
@cython.wraparound(False)
def split_on_whitespace(str document, bint new_line=False, bint tab=False):
    cdef char* c_document
    cdef Py_ssize_t doc_len
    cdef list result = []
    cdef char* start
    cdef char* end
    cdef char* next_char
    cdef int word_len

    c_document = PyUnicode_AsUTF8AndSize(document, &doc_len)
    start = c_document
    end = c_document + doc_len

    while start < end:
        # Skip leading whitespace
        while start < end and (start[0] == b' ' or (new_line and start[0] == b'\n') or (tab and start[0] == b'\t')):
            start += 1

        if start >= end:
            break

        # Find end of word
        next_char = start
        while next_char < end and not (next_char[0] == b' ' or (new_line and next_char[0] == b'\n') or (tab and next_char[0] == b'\t')):
            next_char += 1

        word_len = next_char - start
        if word_len > 0:
            result.append(PyBytes_FromStringAndSize(start, word_len))

        start = next_char

    return result
