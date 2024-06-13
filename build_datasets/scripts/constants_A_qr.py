single_q_tail = " Answer the question using a single word or phrase."
brief_good = "The quality of the evaluated image is quite good without any distortion."

qr_list = [
    [
        [
            "What are the primary degradation(s) observed in the evaluated image?",
            "What is the primary ONE degradation observed in the evaluated image?",
            "What are the primary TWO degradations observed in the evaluated image?",
        ],
        "The primary {ONE} degradation in the evaluated image is {}.",
        "The primary {TWO} degradations in the evaluated image are {}.",
    ],
    [
        [
            "What distortion(s) are most apparent in the evaluated image?",
            "What ONE distortion is most apparent in the evaluated image?",
            "What TWO distortions are most apparent in the evaluated image?",
        ],
        "The most apparent {ONE} distortion in the evaluated image is {}.",
        "The most apparent {TWO} distortions in the evaluated image are {}.",
    ],
    [
        [
            "Identify the chief degradation(s) in the evaluated image.",
            "Identify the chief ONE degradation in the evaluated image.",
            "Identify the chief TWO degradations in the evaluated image.",
        ],
        "The chief {ONE} degradation in the evaluated image is {}.",
        "The chief {TWO} degradations in the evaluated image are {}.",
    ],
    [
        [
            "Pinpoint the foremost image quality issue(s) in the evaluated image.",
            "Pinpoint the foremost ONE image quality issue in the evaluated image.",
            "Pinpoint the foremost TWO image quality issues in the evaluated image.",
        ],
        "The foremost {ONE} image quality issue is {}.",
        "The foremost {TWO} image quality issues are {}.",
    ],
    [
        [
            "What distortion(s) stand out in the evaluated image?",
            "What ONE distortion stands out in the evaluated image?",
            "What TWO distortions stand out in the evaluated image?",
        ],
        "The {ONE} distortion that stands out is {}.",
        "The {TWO} distortions that stand out are {}.",
    ],
    [
        [
            "What distortion(s) are most prominent in the evaluated image?",
            "What ONE distortion is most prominent in the evaluated image?",
            "What TWO distortions are most prominent in the evaluated image?",
        ],
        "The most prominent {ONE} distortion is {}.",
        "The most prominent {TWO} distortions are {}.",
    ],
    [
        [
            "What critical quality degradation(s) are present in the evaluated image?",
            "What critical ONE quality degradation is present in the evaluated image?",
            "What critical TWO quality degradations are present in the evaluated image?",
        ],
        "The critical {ONE} quality degradation presented is {}.",
        "The critical {TWO} quality degradations presented are {}.",
    ],
    [
        [
            "Highlight the most significant distortion(s) in the evaluated image.",
            "Highlight the most significant ONE distortion in the evaluated image.",
            "Highlight the most significant TWO distortions in the evaluated image.",
        ],
        "The most significant {ONE} distortion in the evaluated image is {}.",
        "The most significant {TWO} distortions in the evaluated image are {}.",
    ],
    [
        [
            "What distortion(s) most detrimentally affect the overall quality of the evaluated image?",
            "What ONE distortion most detrimentally affects the overall quality of the evaluated image?",
            "What TWO distortions most detrimentally affect the overall quality of the evaluated image?",
        ],
        "The {ONE} distortion that most detrimentally affects the overall quality is {}.",
        "The {TWO} distortions that most detrimentally affect the overall quality are {}.",
    ],
    [
        [
            "Determine the most impactful distortion(s) in the evaluated image.",
            "Determine the most impactful ONE distortion in the evaluated image.",
            "Determine the most impactful TWO distortions in the evaluated image.",
        ],
        "The most impactful {ONE} distortion in the evaluated image is {}.",
        "The most impactful {TWO} distortions in the evaluated image are {}.",
    ],
    [
        [
            "Identify the most notable distortion(s) in the evaluated image's quality.",
            "Identify the most notable ONE distortion in the evaluated image's quality.",
            "Identify the most notable TWO distortions in the evaluated image's quality.",
        ],
        "The most notable {ONE} distortion in the evaluated image's quality is {}.",
        "The most notable {TWO} distortions in the evaluated image's quality are {}.",
    ],
    [
        [
            "What distortion(s) most significantly affect the evaluated image?",
            "What ONE distortion most significantly affects the evaluated image?",
            "What TWO distortions most significantly affect the evaluated image?",
        ],
        "The {ONE} distortion that most significantly affects the evaluated image is {}.",
        "The {TWO} distortions that most significantly affect the evaluated image are {}.",
    ],
    [
        [
            "Determine the leading degradation(s) in the evaluated image.",
            "Determine the leading ONE degradation in the evaluated image.",
            "Determine the leading TWO degradations in the evaluated image.",
        ],
        "The leading {ONE} degradation is {}.",
        "The leading {TWO} degradations are {}.",
    ],
    [
        [
            "What distortion(s) are most prominent when examining the evaluated image?",
            "What ONE distortion is most prominent when examining the evaluated image?",
            "What TWO distortions are most prominent when examining the evaluated image?",
        ],
        "The most prominent {ONE} distortion is {}.",
        "The most prominent {TWO} distortions are {}.",
    ],
    [
        [
            "What distortion(s) are most evident in the evaluated image?",
            "What ONE distortion is most evident in the evaluated image?",
            "What TWO distortions are most evident in the evaluated image?",
        ],
        "The most evident {ONE} distortion in the evaluated image is {}.",
        "The most evident {TWO} distortions in the evaluated image are {}.",
    ],
    [
        [
            "What quality degradation(s) are most apparent in the evaluated image?",
            "What ONE quality degradation is most apparent in the evaluated image?",
            "What TWO quality degradations are most apparent in the evaluated image?",
        ],
        "The most apparent {ONE} quality degradation is {}.",
        "The most apparent {TWO} quality degradations are {}.",
    ],
    [
        [
            "In terms of image quality, what are the most glaring issue(s) with the evaluated image?",
            "In terms of image quality, what is the most glaring ONE issue with the evaluated image?",
            "In terms of image quality, what are the most glaring TWO issues with the evaluated image?",
        ],
        "The most glaring {ONE} issue with the evaluated image is {}.",
        "The most glaring {TWO} issues with the evaluated image are {}.",
    ],
    [
        [
            "What are the foremost distortion(s) affecting the evaluated image's quality?",
            "What is the foremost ONE distortion affecting the evaluated image's quality?",
            "What are the foremost TWO distortions affecting the evaluated image's quality?",
        ],
        "The foremost {ONE} distortion affecting the evaluated image's quality is {}.",
        "The foremost {TWO} distortions affecting the evaluated image's quality are {}.",
    ],
    [
        [
            "Identify the most critical distortion(s) in the evaluated image.",
            "Identify the most critical ONE distortion in the evaluated image.",
            "Identify the most critical TWO distortions in the evaluated image.",
        ],
        "The most critical {ONE} distortion is {}.",
        "The most critical {TWO} distortions are {}.",
    ],
    [
        [
            "In the evaluated image, what distortion(s) are most detrimental to image quality?",
            "In the evaluated image, what ONE distortion is most detrimental to image quality?",
            "In the evaluated image, what TWO distortions are most detrimental to image quality?",
        ],
        "In the evaluated image, {} is the most detrimental {ONE} distortion to image quality.",
        "In the evaluated image, {} are the most detrimental {TWO} distortions to image quality.",
    ],
    [
        [
            "What are the most severe degradation(s) observed in the evaluated image?",
            "What is the most severe ONE degradation observed in the evaluated image?",
            "What are the most severe TWO degradations observed in the evaluated image?",
        ],
        "The most severe {ONE} degradation is {}.",
        "The most severe {TWO} degradations are {}.",
    ],
    [
        [
            "What are the leading distortion(s) in the evaluated image?",
            "What is the leading ONE distortion in the evaluated image?",
            "What are the leading TWO distortions in the evaluated image?",
        ],
        "The leading {ONE} distortion in the evaluated image is {}.",
        "The leading {TWO} distortions in the evaluated image are {}.",
    ],
    [
        [
            "What are the most critical image quality issue(s) in the evaluated image?",
            "What is the most critical ONE image quality issue in the evaluated image?",
            "What are the most critical TWO image quality issues in the evaluated image?",
        ],
        "The most critical {ONE} image quality issue in the evaluated image is {}.",
        "The most critical {TWO} image quality issues in the evaluated image are {}.",
    ],
    [
        [
            "What distortion(s) most notably affect the clarity of the evaluated image?",
            "What ONE distortion most notably affects the clarity of the evaluated image?",
            "What TWO distortions most notably affect the clarity of the evaluated image?",
        ],
        "The {ONE} distortion that most notably affects the clarity is {}.",
        "The {TWO} distortions that most notably affect the clarity are {}.",
    ],
]


if __name__ == "__main__":
    print(len(qr_list))
