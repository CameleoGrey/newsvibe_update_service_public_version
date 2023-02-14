
import pandas as pd
import pandas.testing as pd_testing
import unittest

from src.classes.paths_config import *
from src.classes.profittm.DataBaseManager import DataBaseManager

class TestDataBaseManager(unittest.TestCase):
    
    def setUp(self):
        self.database_manager = DataBaseManager( Path(interim_dir, "test_db.db") ) 
        
        self.parsed_data_1 = [["a.com", "a", "aaaaaaaaaaaaaaa", None, "12.12.1999"],
                              ["b.com", "b", "bbbbbbbbbbbbbbb", None, "13.12.2000"]]
        self.parsed_data_2 = [["c.com", "c", "ccccccccccccccc", None, "14.12.1999"],
                              ["d.com", "b", "ddddddddddddddd", None, "15.12.2000"]]
        
        self.parsed_data_1 = pd.DataFrame( self.parsed_data_1, columns=["url", "news_source", "content", "article_datetime", "parse_datetime"] )
        self.parsed_data_2 = pd.DataFrame( self.parsed_data_2, columns=["url", "news_source", "content", "article_datetime", "parse_datetime"] )
        
        self.trash_urls_1 = ["aa.com", "bb.com"]
        self.trash_urls_2 = ["cc.com", "dd.com"]
        
        pass
    
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e
    
    def test_delete_tables(self):
        self.database_manager.delete_all_parsed_data_()
        self.database_manager.delete_trash_urls_()
        
        existed_data = self.database_manager.get_parsed_data()
        existed_trash_urls = self.database_manager.get_trash_only_urls()
        existed_urls_to_ignore = self.database_manager.get_urls_to_ignore()
        
        self.assertDataframeEqual(existed_data, 
                                  pd.DataFrame(columns=["url", "news_source", "content", "article_datetime", "parse_datetime"]),
                                  "Deleted parsed data is not equal to empty DataFrame")
        
        self.assertEqual(existed_trash_urls, [], "Deleted trash urls are not equal to empty list")
        self.assertEqual(existed_urls_to_ignore, [], "Deleted ignored urls are not equal to empty list")
        
        pass
    
    def test_update_empty_urls(self):
        
        self.database_manager.update_trash_only_urls(self.trash_urls_1)
        existed_trash_urls = self.database_manager.get_trash_only_urls()
        existed_urls_to_ignore = self.database_manager.get_urls_to_ignore()
        
        self.assertEqual(existed_trash_urls, self.trash_urls_1, "Updated empty trash urls are not equal to trash_urls_1")
        self.assertEqual(existed_urls_to_ignore, self.trash_urls_1, "Updated empty ignored urls are not equal to trash_urls_1")
        
        self.database_manager.delete_all_parsed_data_()
        self.database_manager.delete_trash_urls_()
        
        pass
    
    def test_update_parsed_data(self):
        
        self.database_manager.update_parsed_data(self.parsed_data_1)
        existed_data = self.database_manager.get_parsed_data()
        self.assertDataframeEqual(existed_data, self.parsed_data_1, "Updated empty parsed data is not equal to self.parsed_data_1")
        
        
        self.database_manager.update_parsed_data(self.parsed_data_2)
        existed_data = self.database_manager.get_parsed_data()
        concatenated_parsed_data = pd.concat([self.parsed_data_1, self.parsed_data_2]).reset_index(drop=True)
        self.assertDataframeEqual(existed_data, concatenated_parsed_data, "Updated parsed data is not equal to concatenated parsed_data_1 and parsed_data_2")
        
        self.database_manager.update_trash_only_urls(self.trash_urls_1)
        self.database_manager.update_trash_only_urls(self.trash_urls_2)
        
        existed_urls_to_ignore = self.database_manager.get_urls_to_ignore()
        sample_ignored_urls = []
        sample_ignored_urls += self.trash_urls_1
        sample_ignored_urls += self.trash_urls_2
        sample_ignored_urls += list(self.parsed_data_1["url"].to_numpy())
        sample_ignored_urls += list(self.parsed_data_2["url"].to_numpy())
        existed_urls_to_ignore = list(sorted(existed_urls_to_ignore))
        sample_ignored_urls = list(sorted(sample_ignored_urls))
        self.assertEqual(existed_urls_to_ignore, sample_ignored_urls, "Ignored urls are equal to concatenation of trash and already parsed urls")
        
        pass
    
if __name__ == "__main__":
    unittest.main()
    print("done")