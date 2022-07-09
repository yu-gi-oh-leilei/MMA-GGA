def maxp(logs):
    birth = [0 for i in range(101)]
    death = [0 for i in range(101)]
    for i in range(len(logs)):
        birth[logs[i][0]-1950]=birth[logs[i][0]-1950]+1
        death[logs[i][1]-1950]=death[logs[i][0]-1950]+1
    count = [0 for i in range(101)]
    count[0] = birth[0]-death[0]
    for i in range(1,101):
        count[i]=count[i-1]+birth[i]-death[i]
    
    maxnum = count.index(max(count))+1950
    return maxnum

print(maxp([[1950,2049],[2050,2050]]))


def solution(nums):
    current= min(nums)
    sumpp = sum(nums)
    maxpp = current*sumpp
    j = len(nums)
    i = 0
    while i+1 < j:
        if nums[i]==current:
            current1 = min(nums[i+1:j])
        else:
            current1 = current
        if nums[j-1]==current:
            current2 = min(nums[i:j-1])
        else:
            current2 = current
        sumpp2 = sumpp - nums[j-1]
        sumpp1 = sumpp - nums[i]
        if sumpp2*current2 > sumpp1*current1:
            sumpp=sumpp2
            current=current2
            j=j-1
            if maxpp<sumpp*current:
                maxpp = sumpp*current
        else:
            sumpp=sumpp1
            current=current1
            i=i+1
            if maxpp<sumpp*current:
                print(current)
                maxpp = sumpp*current
    return maxpp%(pow(10,9)+7)
print(solution([3, 2, 5]))
            
            
