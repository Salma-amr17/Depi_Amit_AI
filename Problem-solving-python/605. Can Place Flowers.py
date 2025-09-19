class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        counter = 0
        sum = 0
        sums = []
        ans = 0
        for i in flowerbed:
            last_sum = sum
            sum += (i == 0)
            if sum == last_sum and last_sum > 0:
                sums.append(sum)
                sum = 0

        if sum > 0:
            sums.append(sum)
        for i in sums:
            counter += 1
            if counter  == 1 or counter == len(sums):
                if counter == 1 and flowerbed[0] == 0:
                    ans += i // 2
                elif counter == len(sums) and flowerbed[-1] == 0:
                    ans += i // 2
                else:
                    if i % 2 == 1:
                        ans += i // 2
                    else:
                        ans += i // 2 - 1
            else:
                ans += (i - 1) // 2

        if flowerbed.count(1) == 0:
            ans = (len(flowerbed) + 1) // 2
        return ans >= n