class Solution(object):
    def mergeAlternately(self, word1, word2):
        ans = ''
        sz = min(len(word1), len(word2))
        for i in range(sz):
            ans += word1[i] + word2[i]
        for i in range(sz, max(len(word1), len(word2))):
            if len(word1) > len(word2):
                ans += word1[i]
            else:
                ans += word2[i]

        return ans