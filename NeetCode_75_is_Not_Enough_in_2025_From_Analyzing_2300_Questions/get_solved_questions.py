import json

import requests


def get_leetcode_solved_questions():
    url = 'https://leetcode.com/graphql'

    # GraphQL query to get all problems
    query = '''
    query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
      problemsetQuestionList: questionList(
        categorySlug: $categorySlug
        limit: $limit
        skip: $skip
        filters: $filters
      ) {
        total: totalNum
        questions: data {
          questionId
          questionFrontendId
          title
          titleSlug
          status
          difficulty
        }
      }
    }
    '''

    variables = {
        "categorySlug": "",
        "skip": 0,
        "limit": 5000,  # Adjust this if needed to get all problems
        "filters": {"status": "AC"}
    }

    headers = {
        'Content-Type': 'application/json',
        'Referer': 'https://leetcode.com/problemset/all/',
        # TODO Add your LeetCode session cookie here
        'Cookie': 'LEETCODE_SESSION='
    }

    response = requests.post(url, json={'query': query, 'variables': variables}, headers=headers)
    data = response.json()

    # save result to disk
    with open('../stats/solved_problems.json', 'w') as outfile:
        json.dump(data, outfile)


get_leetcode_solved_questions()
