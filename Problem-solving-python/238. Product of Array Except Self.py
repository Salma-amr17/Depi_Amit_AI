from collections import Counter

class Solution(object):
    def productExceptSelf(self, nums):
        mp = Counter(nums)
        ans = []
        n = len(nums)

        for i in range(n):
            mp[nums[i]] -= 1
            num = 1
            for j in range(-30, 31):
                a = j
                b = mp[a]
                num *= (a ** b)
            ans.append(num)
            mp[nums[i]] += 1

        return ans