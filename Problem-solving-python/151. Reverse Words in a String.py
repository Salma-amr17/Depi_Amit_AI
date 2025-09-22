class Solution(object):
    def reverseWords(self, s):
        ans = s.split()
        res = ''
        ans.reverse()
        for i in range (len(ans)):
            if i != len(ans) - 1:
                res += ans[i] + ' '
            else:
                res += ans[i]
        return res