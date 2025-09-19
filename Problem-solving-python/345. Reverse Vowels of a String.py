class Solution(object):
    def reverseVowels(self, s):
        vowels = ''
        ans = ''
        v = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
        for i in s:
            if i in v:
                vowels += i

        index = 0
        vowels = vowels[::-1]
        for i in s:
            if i not in v:
                ans += i
            else:
                ans += vowels[index]
                index += 1
        return ans