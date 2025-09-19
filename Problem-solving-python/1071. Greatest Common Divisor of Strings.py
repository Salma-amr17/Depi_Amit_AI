class Solution(object):
    def gcdOfStrings(self, str1, str2):
        ans = ''
        small_string = str1
        large_string = str2
        if len(str2) < len(str1):
            small_string = str2
            large_string = str1
        for i in range(len(small_string)):
            new_string = small_string[0 : i + 1]
            if self.ok(large_string, small_string, new_string):
                ans = new_string
        return ans

    def ok(self, large_string, small_string, new_string):
        if len(large_string) % len(new_string) != 0 or len(small_string) % len(new_string) != 0:
            return False
        else:
            counter = 0
            OK = True
            while True:
                s = large_string[counter : counter + len(new_string)]
                if counter + len(new_string) <= len(small_string):
                    s2 = small_string[counter : counter + len(new_string)]
                    if s2 != new_string:
                        OK = False
                if s != new_string:
                    OK = False
                counter += len(new_string)
                if counter >= len(large_string):
                    break
            return OK
    