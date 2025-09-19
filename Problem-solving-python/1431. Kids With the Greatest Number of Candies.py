class Solution(object):
    def kidsWithCandies(self, candies, extraCandies):
        mx = max(candies)
        ans = []
        for i in candies:
            num = i + extraCandies
            if i == mx:
                ans.append(True)
            else:
                if num >= mx:
                    ans.append(True)
                else:
                    ans.append(False)
        return ans