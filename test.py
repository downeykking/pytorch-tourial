

def generate_min_pad(x):
    # x = ["a", "b", "c"] is a list of token
    min_len = 6
    if len(x) < min_len:
        x += ['<pad>'] * (min_len - len(x))
    return x


x = ['This', 'film', 'is', 'terrible']

a = generate_min_pad(x)
print(a)
