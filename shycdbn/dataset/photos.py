# Copyright 2016 Sanghoon Yoon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training Data Preparator"""
from pymongo import MongoClient
from multiprocessing.dummy import Pool
import os
import urllib


class FoursqaurePhotos:
    def __init__(self, host='ambiance.kaist.ac.kr', port=27017, num_threads=16):
        client = MongoClient('mongodb://{}:{}'.format(host, port))
        # 'foursquare/venues'
        db = client.foursquare
        self.venues_collection = db.venues

        # All subcategories of Food and Nightlife spot
        self.categories = ['503288ae91d4c4b30a586d67', '4bf58dd8d48988d10a941735', '4bf58dd8d48988d1c8941735',
                           '4bf58dd8d48988d157941735', '4bf58dd8d48988d14e941735', '56aa371be4b08b9a8d573568',
                           '52e81612bcbc57f1066b7a03', '52af3a5e3cf9994f4e043bea', '52af3a723cf9994f4e043bec',
                           '52af3a7c3cf9994f4e043bed', '52af3a673cf9994f4e043beb', '52af3a903cf9994f4e043bee',
                           '4bf58dd8d48988d1f5931735', '52af3a9f3cf9994f4e043bef', '52af3aaa3cf9994f4e043bf0',
                           '52af3ab53cf9994f4e043bf1', '52af3abe3cf9994f4e043bf2', '52af3ac83cf9994f4e043bf3',
                           '52af3ad23cf9994f4e043bf4', '52af3add3cf9994f4e043bf5', '52af3af23cf9994f4e043bf7',
                           '52af3ae63cf9994f4e043bf6', '52af3afc3cf9994f4e043bf8', '52af3b053cf9994f4e043bf9',
                           '52af3b213cf9994f4e043bfa', '52af3b293cf9994f4e043bfb', '52af3b343cf9994f4e043bfc',
                           '52af3b3b3cf9994f4e043bfd', '52af3b463cf9994f4e043bfe', '52af3b633cf9994f4e043c01',
                           '52af3b513cf9994f4e043bff', '52af3b593cf9994f4e043c00', '52af3b6e3cf9994f4e043c02',
                           '52af3b773cf9994f4e043c03', '52af3b813cf9994f4e043c04', '52af3b893cf9994f4e043c05',
                           '52af3b913cf9994f4e043c06', '52af3b9a3cf9994f4e043c07', '52af3ba23cf9994f4e043c08',
                           '4bf58dd8d48988d145941735', '4eb1bd1c3b7b55596b4a748f', '52e81612bcbc57f1066b79fb',
                           '52af0bd33cf9994f4e043bdd', '52960eda3cf9994f4e043ac9', '52960eda3cf9994f4e043acb',
                           '52960eda3cf9994f4e043aca', '52960eda3cf9994f4e043acc', '52960eda3cf9994f4e043ac7',
                           '52960eda3cf9994f4e043ac8', '52960eda3cf9994f4e043ac5', '52960eda3cf9994f4e043ac6',
                           '4deefc054765f83613cdba6f', '55a59bace4b013909087cb0c', '55a59bace4b013909087cb30',
                           '55a59bace4b013909087cb21', '55a59bace4b013909087cb06', '55a59bace4b013909087cb1b',
                           '55a59bace4b013909087cb1e', '55a59bace4b013909087cb18', '55a59bace4b013909087cb24',
                           '55a59bace4b013909087cb15', '55a59bace4b013909087cb27', '55a59bace4b013909087cb12',
                           '4bf58dd8d48988d1d2941735', '55a59bace4b013909087cb2d', '55a59a31e4b013909087cb00',
                           '55a59af1e4b013909087cb03', '55a59bace4b013909087cb2a', '55a59bace4b013909087cb0f',
                           '55a59bace4b013909087cb33', '55a59bace4b013909087cb09', '55a59bace4b013909087cb36',
                           '4bf58dd8d48988d111941735', '56aa371be4b08b9a8d5734e4', '56aa371be4b08b9a8d5734f0',
                           '56aa371be4b08b9a8d5734e7', '56aa371be4b08b9a8d5734ed', '56aa371be4b08b9a8d5734ea',
                           '4bf58dd8d48988d113941735', '4bf58dd8d48988d156941735', '4eb1d5724b900d56c88a45fe',
                           '4bf58dd8d48988d1d1941735', '56aa371be4b08b9a8d57350e', '56aa371be4b08b9a8d573502',
                           '4bf58dd8d48988d149941735', '52af39fb3cf9994f4e043be9', '4bf58dd8d48988d14a941735',
                           '4bf58dd8d48988d142941735', '4bf58dd8d48988d169941735', '52e81612bcbc57f1066b7a01',
                           '4bf58dd8d48988d1df931735', '4bf58dd8d48988d179941735', '4bf58dd8d48988d16a941735',
                           '52e81612bcbc57f1066b7a02', '52e81612bcbc57f1066b79f1', '4bf58dd8d48988d143941735',
                           '52e81612bcbc57f1066b7a0c', '52e81612bcbc57f1066b79f4', '4bf58dd8d48988d16c941735',
                           '4bf58dd8d48988d128941735', '4bf58dd8d48988d16d941735', '4bf58dd8d48988d17a941735',
                           '4bf58dd8d48988d154941735', '4bf58dd8d48988d144941735', '5293a7d53cf9994f4e043a45',
                           '4bf58dd8d48988d1e0931735', '52e81612bcbc57f1066b7a00', '52e81612bcbc57f1066b79f2',
                           '52f2ae52bcbc57f1066b8b81', '4bf58dd8d48988d146941735', '4bf58dd8d48988d1bc941735',
                           '512e7cae91d4cbb4e5efe0af', '4bf58dd8d48988d1c9941735', '5744ccdfe4b0c0459246b4e2',
                           '52e81612bcbc57f1066b7a0a', '4bf58dd8d48988d1d0941735', '4bf58dd8d48988d147941735',
                           '4bf58dd8d48988d148941735', '4bf58dd8d48988d108941735', '5744ccdfe4b0c0459246b4d0',
                           '52e928d0bcbc57f1066b7e97', '56aa371be4b08b9a8d5734f3', '52960bac3cf9994f4e043ac4',
                           '52e928d0bcbc57f1066b7e98', '4bf58dd8d48988d109941735', '52e81612bcbc57f1066b7a05',
                           '4bf58dd8d48988d10b941735', '4bf58dd8d48988d16e941735', '4edd64a0c7ddd24ca188df1a',
                           '52e81612bcbc57f1066b7a09', '4bf58dd8d48988d120951735', '56aa371be4b08b9a8d57350b',
                           '4bf58dd8d48988d1cb941735', '57558b36e4b065ecebd306b6', '57558b36e4b065ecebd306b8',
                           '57558b36e4b065ecebd306bc', '57558b36e4b065ecebd306b0', '57558b36e4b065ecebd306c5',
                           '57558b36e4b065ecebd306c0', '57558b36e4b065ecebd306cb', '57558b36e4b065ecebd306ce',
                           '57558b36e4b065ecebd306d1', '57558b36e4b065ecebd306b4', '57558b36e4b065ecebd306b2',
                           '57558b35e4b065ecebd306ad', '57558b36e4b065ecebd306d4', '57558b36e4b065ecebd306d7',
                           '57558b36e4b065ecebd306da', '57558b36e4b065ecebd306ba', '4bf58dd8d48988d10c941735',
                           '4d4ae6fc7a7b7dea34424761', '55d25775498e9f6a0816a37a', '4bf58dd8d48988d155941735',
                           '56aa371ce4b08b9a8d573583', '56aa371ce4b08b9a8d573572', '56aa371ce4b08b9a8d57358e',
                           '56aa371ce4b08b9a8d57358b', '56aa371ce4b08b9a8d573574', '56aa371ce4b08b9a8d573592',
                           '56aa371ce4b08b9a8d573578', '56aa371ce4b08b9a8d57357b', '56aa371ce4b08b9a8d573587',
                           '56aa371ce4b08b9a8d57357f', '56aa371ce4b08b9a8d573576', '4bf58dd8d48988d10d941735',
                           '4c2cd86ed066bed06c3c5209', '53d6c1b0e4b02351e88a83e8', '53d6c1b0e4b02351e88a83e2',
                           '53d6c1b0e4b02351e88a83d8', '53d6c1b0e4b02351e88a83d6', '53d6c1b0e4b02351e88a83e6',
                           '53d6c1b0e4b02351e88a83e4', '53d6c1b0e4b02351e88a83da', '53d6c1b0e4b02351e88a83d4',
                           '53d6c1b0e4b02351e88a83dc', '53d6c1b0e4b02351e88a83e0', '52e81612bcbc57f1066b79f3',
                           '53d6c1b0e4b02351e88a83d2', '53d6c1b0e4b02351e88a83de', '4bf58dd8d48988d10e941735',
                           '52e81612bcbc57f1066b79ff', '52e81612bcbc57f1066b79fe', '4bf58dd8d48988d16f941735',
                           '52e81612bcbc57f1066b79fa', '54135bf5e4b08f3d2429dfe5', '54135bf5e4b08f3d2429dff3',
                           '54135bf5e4b08f3d2429dff5', '54135bf5e4b08f3d2429dfe2', '54135bf5e4b08f3d2429dff2',
                           '54135bf5e4b08f3d2429dfe1', '54135bf5e4b08f3d2429dfe3', '54135bf5e4b08f3d2429dfe8',
                           '54135bf5e4b08f3d2429dfe9', '54135bf5e4b08f3d2429dfe6', '54135bf5e4b08f3d2429dfdf',
                           '54135bf5e4b08f3d2429dfe4', '54135bf5e4b08f3d2429dfe7', '54135bf5e4b08f3d2429dfea',
                           '54135bf5e4b08f3d2429dfeb', '54135bf5e4b08f3d2429dfed', '54135bf5e4b08f3d2429dfee',
                           '54135bf5e4b08f3d2429dff4', '54135bf5e4b08f3d2429dfe0', '54135bf5e4b08f3d2429dfdd',
                           '54135bf5e4b08f3d2429dff6', '54135bf5e4b08f3d2429dfef', '54135bf5e4b08f3d2429dff0',
                           '54135bf5e4b08f3d2429dff1', '54135bf5e4b08f3d2429dfde', '54135bf5e4b08f3d2429dfec',
                           '4bf58dd8d48988d10f941735', '52e81612bcbc57f1066b7a06', '55a5a1ebe4b013909087cbb6',
                           '55a5a1ebe4b013909087cb7c', '55a5a1ebe4b013909087cba7', '55a5a1ebe4b013909087cba1',
                           '55a5a1ebe4b013909087cba4', '55a5a1ebe4b013909087cb95', '55a5a1ebe4b013909087cb89',
                           '55a5a1ebe4b013909087cb9b', '55a5a1ebe4b013909087cb98', '55a5a1ebe4b013909087cbbf',
                           '55a5a1ebe4b013909087cb79', '55a5a1ebe4b013909087cbb0', '55a5a1ebe4b013909087cbb3',
                           '55a5a1ebe4b013909087cb74', '55a5a1ebe4b013909087cbaa', '55a5a1ebe4b013909087cb83',
                           '55a5a1ebe4b013909087cb8c', '55a5a1ebe4b013909087cb92', '55a5a1ebe4b013909087cb8f',
                           '55a5a1ebe4b013909087cb86', '55a5a1ebe4b013909087cbb9', '55a5a1ebe4b013909087cb7f',
                           '55a5a1ebe4b013909087cbbc', '55a5a1ebe4b013909087cb9e', '55a5a1ebe4b013909087cbc2',
                           '55a5a1ebe4b013909087cbad', '4bf58dd8d48988d110941735', '52e81612bcbc57f1066b79fc',
                           '52e81612bcbc57f1066b79fd', '4bf58dd8d48988d112941735', '5283c7b4e4b094cb91ec88d7',
                           '4bf58dd8d48988d152941735', '52939a8c3cf9994f4e043a35', '5745c7ac498e5d0483112fdb',
                           '4bf58dd8d48988d107941735', '5294c7523cf9994f4e043a62', '52939ae13cf9994f4e043a3b',
                           '52939a9e3cf9994f4e043a36', '52939a643cf9994f4e043a33', '5294c55c3cf9994f4e043a61',
                           '52939af83cf9994f4e043a3d', '52939aed3cf9994f4e043a3c', '52939aae3cf9994f4e043a37',
                           '52939ab93cf9994f4e043a38', '5294cbda3cf9994f4e043a63', '52939ac53cf9994f4e043a39',
                           '52939ad03cf9994f4e043a3a', '52939a7d3cf9994f4e043a34', '4bf58dd8d48988d16b941735',
                           '4eb1bfa43b7b52c0e1adc2e8', '56aa371be4b08b9a8d573558', '4bf58dd8d48988d1cd941735',
                           '4bf58dd8d48988d1be941735', '4bf58dd8d48988d1bf941735', '4bf58dd8d48988d1c3941735',
                           '4bf58dd8d48988d1c0941735', '4bf58dd8d48988d153941735', '4bf58dd8d48988d151941735',
                           '56aa371ae4b08b9a8d5734ba', '5744ccdfe4b0c0459246b4d3', '4bf58dd8d48988d1c1941735',
                           '56aa371be4b08b9a8d573529', '5744ccdfe4b0c0459246b4ca', '5744ccdfe4b0c0459246b4a8',
                           '52e81612bcbc57f1066b79f7', '4bf58dd8d48988d115941735', '52e81612bcbc57f1066b79f9',
                           '4bf58dd8d48988d1c2941735', '52e81612bcbc57f1066b79f8', '56aa371be4b08b9a8d573508',
                           '4bf58dd8d48988d1ca941735', '52e81612bcbc57f1066b7a04', '4def73e84765ae376e57713a',
                           '56aa371be4b08b9a8d5734c7', '4bf58dd8d48988d1c4941735', '52e928d0bcbc57f1066b7e9d',
                           '52e928d0bcbc57f1066b7e9c', '5293a7563cf9994f4e043a44', '4bf58dd8d48988d1bd941735',
                           '4bf58dd8d48988d1c5941735', '4bf58dd8d48988d1c6941735', '5744ccdde4b0c0459246b4a3',
                           '4bf58dd8d48988d1ce941735', '56aa371be4b08b9a8d57355a', '4bf58dd8d48988d1c7941735',
                           '4bf58dd8d48988d1dd931735', '4bf58dd8d48988d14f941735', '4bf58dd8d48988d14d941735',
                           '4bf58dd8d48988d1db931735', '4bf58dd8d48988d150941735', '5413605de4b0ae91d18581a9',
                           '4bf58dd8d48988d1cc941735', '4bf58dd8d48988d158941735', '4bf58dd8d48988d1dc931735',
                           '56aa371be4b08b9a8d573538', '57558b36e4b065ecebd306dd', '530faca9bcbc57f1066bc2f3',
                           '530faca9bcbc57f1066bc2f4', '5283c7b4e4b094cb91ec88d8', '5283c7b4e4b094cb91ec88d9',
                           '5283c7b4e4b094cb91ec88db', '5283c7b4e4b094cb91ec88d6', '56aa371be4b08b9a8d573535',
                           '56aa371be4b08b9a8d5734bd', '5283c7b4e4b094cb91ec88d5', '5283c7b4e4b094cb91ec88da',
                           '530faca9bcbc57f1066bc2f2', '56aa371be4b08b9a8d5734bf', '56aa371be4b08b9a8d5734c1',
                           '5283c7b4e4b094cb91ec88d4', '4f04af1f2fb6e1c99f3db0bb', '52e928d0bcbc57f1066b7e9a',
                           '52e928d0bcbc57f1066b7e9b', '52e928d0bcbc57f1066b7e96', '4bf58dd8d48988d1d3941735',
                           '4bf58dd8d48988d14c941735', '4d4b7105d754a06374d81259', '52e81612bcbc57f1066b7a0d',
                           '56aa371ce4b08b9a8d57356c', '4bf58dd8d48988d117941735', '52e81612bcbc57f1066b7a0e',
                           '4bf58dd8d48988d11e941735', '4bf58dd8d48988d118941735', '4bf58dd8d48988d1d8941735',
                           '4bf58dd8d48988d119941735', '4bf58dd8d48988d1d5941735', '4bf58dd8d48988d120941735',
                           '4bf58dd8d48988d11b941735', '4bf58dd8d48988d11c941735', '4bf58dd8d48988d11d941735',
                           '56aa371be4b08b9a8d57354d', '4bf58dd8d48988d122941735', '4bf58dd8d48988d123941735',
                           '4bf58dd8d48988d116941735', '50327c8591d4c4b30a586d5d', '4bf58dd8d48988d121941735',
                           '53e510b7498ebcb1801b55d4', '4bf58dd8d48988d11f941735', '4bf58dd8d48988d11a941735',
                           '4bf58dd8d48988d1d4941735', '4bf58dd8d48988d1d6941735', '4d4b7105d754a06376d81259']

        # Thread pool for
        # The number of threads is set to double of the number of vCPUs.
        self.thread_pool = Pool(num_threads)

    def prepare(self):
        """
        Fetch all photos from venue info
        Each document in collection `'venues'` has `'photos'` field which has all the photos taken at that venue.
        """
        venues = self.venues_collection.find({
            "photos.items": {"$exists": True},
            "categories": {"$elemMatch": {"id": {"$in": self.categories}}}
        })
        print '{} venues fetched.'.format(venues.count())

        """
        [[item1, item2], [item3, item4]]
        """
        list_of_photo_lists = self.thread_pool.map(lambda venue: venue['photos']['items'], venues)

        def merge_lists(list_a, list_b):
            list_a += list_b
            return list_a
        photos = reduce(merge_lists, list_of_photo_lists, [])

        photo_urls = self.thread_pool.map(lambda photo: photo['prefix'] + '300x300' + photo['suffix'], photos)
        print '{} photos.'.format(len(photo_urls))

        # Find or download all the images
        work_dir = 'venue-photo-data'

        def find_or_download(url):
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)
            filename = url.split('/')[-1]
            filepath = os.path.join(work_dir, filename)
            if not os.path.exists(filepath):
                filepath, _ = urllib.urlretrieve(url, filepath)
                statinfo = os.stat(filepath)
                print 'Succesfully downloaded', filename, statinfo.st_size, 'bytes.'
            return filepath
        return self.thread_pool.map(find_or_download, photo_urls)
