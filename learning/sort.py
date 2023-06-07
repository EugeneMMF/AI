import copy


def merge_sort(arr,**kwargs):
    if kwargs.get('indices'):
        indices = kwargs['indices']
    else:
        indices = []
        for i in range(len(arr)):
            indices.append(i)
    length = len(arr)
    if length == 1:
        return [arr,indices]
    mid = int(length/2)
    left_arr = arr[:mid]
    left_indices = indices[:mid]
    right_arr = arr[mid:]
    right_indices = indices[mid:]
    left = merge_sort(left_arr,indices = left_indices)
    left_arr = left[0]
    left_indices = left[1]
    right = merge_sort(right_arr,indices = right_indices)
    right_arr = right[0]
    right_indices = right[1]
    new_arr = []
    new_indices = []
    left_counter = 0
    right_counter = 0
    len_right = len(right[0])
    len_left = len(left[0])
    while len(new_arr) != length:
        if right_counter < len_right:
            if left_counter < len_left:
                if right_arr[right_counter] < left_arr[left_counter]:
                    new_arr.append(right_arr[right_counter])
                    new_indices.append(right_indices[right_counter])
                    right_counter += 1
                else:
                    new_arr.append(left_arr[left_counter])
                    new_indices.append(left_indices[left_counter])
                    left_counter += 1
            else:
                new_arr.append(right_arr[right_counter])
                new_indices.append(right_indices[right_counter])
                right_counter += 1
        else:
            new_arr.append(left_arr[left_counter])
            new_indices.append(left_indices[left_counter])
            left_counter += 1
    return [new_arr,new_indices]

arr = [23,12,13,54,751,2,12,41]
print(merge_sort(arr))