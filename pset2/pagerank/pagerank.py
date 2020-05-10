import os
import random
import re
import sys
import time

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    transition_model = {}
    num_page = len(corpus)

    # recording existing pages from corpus and assign initial probability to each page
    for key in corpus:
        transition_model[key] = (1-damping_factor)/num_page
    
    num_link = len(corpus[page])
    # reiterate link of page to calculate link probability
    for link in corpus[page]:
        transition_model[link] += damping_factor/num_link
    
    return transition_model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # inititalise pagerank
    count = n
    pagerank = {}
    for key in corpus:
        pagerank[key] = 0
    # 1. generate random number which determine the first page to start surfering
    num_page = len(corpus)
    corpus_key = list(corpus)
    rand_key = random.randint(0,num_page-1)
    rand_page = corpus_key[rand_key]
    pagerank[rand_page] = pagerank.get(rand_page,0) + 1

    while True:
    # 2. use transition model to find out probability to the next page
        next_page_model = transition_model(corpus,rand_page,damping_factor)

        link_pages = list(next_page_model)
        pages_prob = list(next_page_model.values())
        
    # 3. generate random number to see which page to go to, then record the amount of time such page has been landed on
        rand_key = random.uniform(0,1)
        for i in range(len(link_pages)):
            if i == 0:
                if rand_key < pages_prob[i]:
                    rand_page = link_pages[i]
                    pagerank[rand_page] = pagerank.get(rand_page,0) + 1
            else:
                if rand_key < (pages_prob[i]+pages_prob[i-1]) and rand_key > (pages_prob[i-1]):
                    rand_page = link_pages[i]
                    pagerank[rand_page] = pagerank.get(rand_page,0) + 1
    
    # 4. repeat step 2 and 3 for n time
        count -= 1
        if (count<1):
            break
    # 5. return pagerank
    for key in pagerank:
        pagerank[key] = pagerank.get(key,0)/n
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # inititalise pagerank
    num_page = len(corpus)
    pagerank = {}
    for key in corpus:
        pagerank[key] = 1/num_page
    
    # iterate with the algorithm until difference is smaller than 0.001
    diff = 1
    while diff > 0.001:
        sum_check = 0
        for key in pagerank:
            check = pagerank[key]
            first = (1 - damping_factor)/num_page
            second = 0
            for link in corpus[key]:
                second += pagerank[link]/len(corpus[link])
            pagerank[key] = first + damping_factor * second

            # check differences
            diff = abs(check - pagerank[key])
            check = pagerank[key]
            sum_check += pagerank[key]
        
        print(pagerank)
        print(sum_check)
        time.sleep(0.5)
        
    return pagerank


if __name__ == "__main__":
    main()
